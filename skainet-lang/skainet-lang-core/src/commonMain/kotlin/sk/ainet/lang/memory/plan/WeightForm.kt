package sk.ainet.lang.memory.plan

import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * The form a weight is asked to take in memory — what it is encoded as, what order its bytes are
 * in, and where they live (#1109).
 *
 * ## Why this is one type and not three flags
 *
 * The three decisions were already being made, separately, by whoever constructed the loader:
 * `QuantPolicy` said whether to dequantize, `StagingPolicy` said heap or mapped, and
 * `WeightOrientation` said which way round the shape was. Three problems with that.
 *
 * Nobody asked the *target*. A weight arrives in whatever the file holds, and nothing consults the
 * backend about which encodings its kernels can actually feed — so a format with no kernel becomes
 * a per-call dequantization that nobody declared, and a format with a good kernel can still be
 * handed bytes in the wrong order.
 *
 * `WeightOrientation` also stops at the shape: it reverses a 2-D weight's dimensions and leaves the
 * bytes alone. The order that decides whether a packed kernel reads the right blocks (#973) had no
 * name in that vocabulary at all. [order] gives it one.
 *
 * And it was the caller's decision when it should have been a resolved one. What the file holds,
 * what the device has and what the plan can afford are all knowable at load time, in one place —
 * asking the user to pick from three enums is asking them to do that resolution by hand, for a
 * device they may not be building for. [WeightFormResolver] does it instead.
 *
 * ## Repacking and dequantizing are allowed
 *
 * Both are legitimate; a dequantization is sometimes exactly right. The point is that they become
 * *stated intent*, executed once at load, rather than accidents discovered per forward pass — and
 * that the cost is visible before it is paid, because a dequantization can quadruple a tensor.
 *
 * @property encoding what the bytes should encode once loaded
 * @property order which way the packed blocks run
 * @property shape which way round the dimensions are labelled
 * @property residency whether the bytes live on the heap or in file-backed pages
 */
public data class WeightForm(
    val encoding: EncodingRequest = EncodingRequest.KeepAsStored,
    val order: WeightByteOrder = WeightByteOrder.AS_STORED,
    val shape: WeightShapeOrientation = WeightShapeOrientation.AS_STORED,
    val residency: WeightResidency = WeightResidency.HEAP,
) {
    /** True when this form asks for nothing — the bytes are used exactly as the file holds them. */
    public val isPassThrough: Boolean
        get() = encoding == EncodingRequest.KeepAsStored &&
            order == WeightByteOrder.AS_STORED &&
            shape == WeightShapeOrientation.AS_STORED &&
            residency == WeightResidency.HEAP

    public companion object {
        /** What every loader did before #1109: the file's bytes, its order, on the heap. */
        public val AS_STORED_ON_HEAP: WeightForm = WeightForm()
    }
}

/** What the loaded bytes should encode. */
public sealed interface EncodingRequest {

    /** Whatever the file holds, untouched. The default, and the only one that costs nothing. */
    public data object KeepAsStored : EncodingRequest

    /**
     * Decode to a dense [dtype] at load.
     *
     * Correct when no kernel can feed the stored encoding — the alternative is dequantizing on
     * every forward pass instead of once — and expensive in the obvious way: a Q4_K tensor becomes
     * roughly eight times its size as FP32. That is why a resolver that chooses this should be able
     * to say so before the load rather than after the OOM.
     */
    public data class DequantizeTo(val dtype: DType) : EncodingRequest

    /**
     * Re-encode to [encoding] at load — a quantization the file did not already have.
     *
     * Distinct from [DequantizeTo] in direction and from [KeepAsStored] in cost: it re-quantizes,
     * so it is lossy, and it is only ever right when the target's kernels want a format the file
     * does not carry.
     */
    public data class RequantizeTo(val encoding: TensorEncoding) : EncodingRequest
}

/**
 * Which order a packed weight's blocks run in.
 *
 * This is the distinction `WeightOrientation` could not express. It reverses *dimensions*; this is
 * about *bytes*, and for a block-quantized weight the two are independent: canonical storage and
 * kernel feed order hold the same blocks in different physical positions, and they coincide only at
 * one block per row (#973, #968).
 */
public enum class WeightByteOrder {

    /** The file's own order — canonical row-major blocks, as every GGUF-shaped producer writes. */
    AS_STORED,

    /**
     * The order the packed matmul kernels read: input-block-major, every output row's block for one
     * input block contiguous.
     *
     * Asking for this at load is what makes the #1096 relayout unnecessary — the weight arrives in
     * feed order and the first forward pass has nothing to convert.
     */
    KERNEL_FEED,
}

/**
 * Which way round a 2-D weight's dimensions are labelled.
 *
 * Orthogonal to [WeightByteOrder], and easy to conflate with it: this is about the *shape*, that is
 * about the *bytes*. Reversing the dimensions of a packed weight moves no data at all, while
 * changing its block order moves every block and leaves the shape alone. A weight can need either,
 * both, or neither.
 */
public enum class WeightShapeOrientation {

    /** The file's own order, unreversed — GGUF `ne`, so `[in, out]` for a 2-D weight. */
    AS_STORED,

    /**
     * Logical `[out, in]`: the convention the engine, HF checkpoints and every kernel assume. The
     * bytes are untouched, because they already are `[out, in]` row-major; only the label changes.
     */
    OUT_IN,
}

/** Where a weight's bytes live. The loader-side spelling of `StagingPolicy` (#1037). */
public enum class WeightResidency {

    /** Read onto the managed heap. The historical behaviour, and the only option in a browser. */
    HEAP,

    /**
     * Serve from file-backed pages, which the OS pages in on demand and evicts under pressure —
     * the difference between fitting a model on a 2 GB device and not (#921, #922).
     */
    MAPPED,
}
