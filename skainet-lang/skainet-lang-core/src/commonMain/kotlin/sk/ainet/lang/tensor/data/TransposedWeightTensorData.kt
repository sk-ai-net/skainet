package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * `Wᵀ` for a weight whose bytes cannot be transposed — the shape is the transpose, the payload is
 * still `W`'s, and nothing may read the payload as if it were `Wᵀ`'s (#1108, following #973).
 *
 * ## Why this exists
 *
 * Block-quantized weights quantize runs along the input dimension, so a real transpose needs
 * requantization. #973 made `transpose` refuse them rather than keep doing the O(bytes) layout
 * conversion it had been doing under transpose's name. That was right about the operation and
 * wrong about the ergonomics: it meant `x.matmul(w.t())` compiled for a dense weight and threw for
 * a packed one, so the code an author writes depended on which file the user loaded and which
 * device it ran on — things the author cannot know.
 *
 * This restores one spelling for both. `transpose` on a packed weight returns this; `matmul`
 * recognises it and asks for the product directly, which is the operation that *is* defined.
 *
 * ## Why this is not the old lazy transpose
 *
 * The transpose that #973 removed was a bare metadata relabel: same bytes, swapped shape, still
 * claiming to be ordinary packed data. Kernels took it at its word and read canonical bytes as
 * input-block-major ones, which is not an error — it is plausible garbage
 * (`NativeLazyTransposeGroundTruthReproTest`, #968, SKaiNET-transformers#307).
 *
 * The difference here is that this **is not** [sk.ainet.lang.tensor.storage.PackedBlockStorage].
 * It has no `packedData`, so no kernel can reach the bytes through it and no `is PackedBlockStorage`
 * check will route it down a packed path by mistake. The only way to get a product out of it is the
 * `matmulWeightTransposed` primitive, which relayouts properly and once.
 *
 * ## What reading it gives you
 *
 * Element access mirrors [weight] with the indices swapped — whatever `weight[j, i]` means,
 * `this[i, j]` means the same thing. For packed data that is the quantized code, exactly as it is
 * on the weight itself; this type invents no new semantics, it only transposes the ones it wraps.
 *
 * @property weight the untransposed weight; `[out, in]` where this is `[in, out]`
 */
public class TransposedWeightTensorData<T : DType, V>(
    public val weight: TensorData<T, V>,
) : TensorData<T, V> {

    init {
        require(weight.shape.rank == 2) {
            "a transposed-weight view is only defined for a rank-2 weight, got ${weight.shape}"
        }
    }

    override val shape: Shape = Shape(weight.shape[1], weight.shape[0])

    /** The wrapped weight's encoding: transposing does not change how the bytes are packed. */
    override val encoding: sk.ainet.lang.tensor.storage.TensorEncoding? get() = weight.encoding

    /**
     * The weight's view with its last two axes swapped, or `null` if the weight has none.
     *
     * Safe where a relabel is not, because [sk.ainet.lang.memory.TensorView] transposes a blocked
     * layout by moving `blockAxis` with the axes (#1034) — the view knows where its blocks run, so
     * decoding through it addresses the right elements.
     */
    override val view: sk.ainet.lang.memory.TensorView? get() = weight.view?.transpose()

    override fun get(vararg indices: Int): V {
        require(indices.size == 2) { "expected 2 indices for shape $shape, got ${indices.size}" }
        return weight[indices[1], indices[0]]
    }

    override fun set(vararg indices: Int, value: V) {
        require(indices.size == 2) { "expected 2 indices for shape $shape, got ${indices.size}" }
        weight[indices[1], indices[0]] = value
    }

    override fun toString(): String = "Wᵀ(${weight::class.simpleName}, $shape, unmaterialized)"
}
