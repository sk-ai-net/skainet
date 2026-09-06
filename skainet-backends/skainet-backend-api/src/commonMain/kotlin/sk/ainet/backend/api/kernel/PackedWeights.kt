package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The engine's own answer to "how do I get a packed weight into the shape a kernel reads"
 * (#973 proposal item 3; #1097).
 *
 * Before this, every converter kept a private copy of the row-major → input-block-major relayout,
 * and they diverged: the census in #973 found one inlined in a downstream Apertus path that had
 * drifted from the shared packer it was copied from, three different conventions applied to the
 * same GGUF K-quant tensor depending on which converter loaded it, and layout knowledge living in
 * the wrong repository entirely. One function, owned here, is the fix — and the fixtures in
 * [PackedLayoutFixtures] are how a downstream repository proves it agrees.
 *
 * The relayout is O(bytes). Call it **once, at load**; a weight prepacked at load hits the packed
 * kernel's key directly and the dispatcher copies nothing per call (#1095).
 */
public object PackedWeights {

    /**
     * [weight] in the order the packed matmul kernels read — [BlockOrder.INPUT_BLOCK_MAJOR].
     *
     * Returns the view unchanged when it is already in that order, so calling this twice is
     * harmless: unlike the packed "transpose" it replaces, it is idempotent by construction.
     *
     * @throws IllegalArgumentException if [weight] is not a 2-D block-packed view
     */
    public fun prepackForMatmul(
        weight: TensorView,
        scope: Scope = Scope.Ambient,
        sink: TraceSink = NoopTraceSink,
    ): TensorView {
        require(weight.layout.blocked) { "prepackForMatmul takes a block-packed weight, got ${weight.format}" }
        require(weight.shape.rank == 2) { "a matmul weight is 2-D [out, in], got ${weight.shape}" }
        requireOutIn(weight.shape[0], weight.shape[1], weight.format.encoding)
        return weight.prepack(BlockOrder.INPUT_BLOCK_MAJOR, scope, sink)
    }

    /** [weight] back in the canonical order a file holds — the inverse of [prepackForMatmul]. */
    public fun toCanonical(
        weight: TensorView,
        scope: Scope = Scope.Ambient,
        sink: TraceSink = NoopTraceSink,
    ): TensorView {
        require(weight.layout.blocked) { "toCanonical takes a block-packed weight, got ${weight.format}" }
        return weight.prepack(BlockOrder.ROW_MAJOR, scope, sink)
    }

    /**
     * The relayout at the byte level, for a converter that holds bytes rather than views:
     * `[out, in]` blocks in canonical order → kernel order.
     *
     * `out[b * rows + o] = in[o * blocksPerRow + b]`, block by block. This is the *only* sanctioned
     * implementation of that permutation; a private copy is what #973 exists to stop.
     */
    public fun toKernelOrder(canonical: ByteArray, rows: Int, blocksPerRow: Int, bytesPerBlock: Int): ByteArray =
        permute(canonical, rows, blocksPerRow, bytesPerBlock, toKernelOrder = true)

    /** The inverse of [toKernelOrder]: kernel-order bytes back to canonical. */
    public fun toCanonicalOrder(kernelOrder: ByteArray, rows: Int, blocksPerRow: Int, bytesPerBlock: Int): ByteArray =
        permute(kernelOrder, rows, blocksPerRow, bytesPerBlock, toKernelOrder = false)

    private fun permute(
        source: ByteArray,
        rows: Int,
        blocksPerRow: Int,
        bytesPerBlock: Int,
        toKernelOrder: Boolean,
    ): ByteArray {
        require(rows > 0 && blocksPerRow > 0 && bytesPerBlock > 0) { "geometry must be positive" }
        val required = rows * blocksPerRow * bytesPerBlock
        require(source.size >= required) {
            "need $required bytes for a $rows × $blocksPerRow block grid of ${bytesPerBlock}-byte blocks, got ${source.size}"
        }
        val out = ByteArray(required)
        for (o in 0 until rows) {
            for (b in 0 until blocksPerRow) {
                val canonical = (o * blocksPerRow + b) * bytesPerBlock
                val kernel = (b * rows + o) * bytesPerBlock
                val from = if (toKernelOrder) canonical else kernel
                val to = if (toKernelOrder) kernel else canonical
                source.copyInto(out, to, from, from + bytesPerBlock)
            }
        }
        return out
    }

    /**
     * Refuse a weight that looks transposed, instead of computing the wrong permutation from it
     * (#973 census contradiction #6; #1098).
     *
     * A packed weight's blocks tile the **input** dimension, so for a correctly oriented
     * `[out, in]` weight `in` is a multiple of the block size. When the *first* dimension is
     * block-aligned and the second is not, the tensor is almost certainly `[in, out]` — the shape
     * a GGUF's `ne` order produces — and relayouting it would silently permute the wrong grid.
     *
     * The check is a heuristic and says so: a square weight, or one where both dimensions are
     * aligned, passes either way. It catches the case that actually shipped.
     */
    public fun requireOutIn(rows: Int, inputDim: Int, encoding: TensorEncoding) {
        val spec = encoding.blockSpec ?: return
        if (spec.isPerTensor) return
        val blockSize = spec.blockSize
        val inputAligned = inputDim % blockSize == 0
        val rowsAligned = rows % blockSize == 0
        require(inputAligned || !rowsAligned) {
            "weight [$rows, $inputDim] looks like [in, out]: ${encoding.name} tiles the *input* dimension in " +
                "blocks of $blockSize, and $inputDim is not a multiple of it while $rows is. A GGUF's ne order " +
                "produces exactly this — load with WeightShapeOrientation.OUT_IN, or transpose the label before " +
                "relayouting (#973, the Packed weight layout page in the docs site (explanation/packed-weight-layout))."
        }
        require(inputAligned) {
            "weight [$rows, $inputDim] cannot be relayouted: ${encoding.name} needs the input dimension to be a " +
                "multiple of $blockSize"
        }
    }

    /** Block geometry of [encoding] — what a converter needs to call [toKernelOrder]. */
    public fun blocksPerRow(encoding: TensorEncoding, inputDim: Int): Int {
        val spec = encoding.blockSpec
            ?: throw IllegalArgumentException("$encoding is not block-structured")
        require(inputDim % spec.blockSize == 0) {
            "${encoding.name} tiles the input dimension in blocks of ${spec.blockSize}; $inputDim is not a multiple"
        }
        return inputDim / spec.blockSize
    }
}
