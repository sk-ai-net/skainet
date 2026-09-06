package sk.ainet.lang.memory

import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.Int8

/** Where an encoding keeps the scale that turns a stored code back into a value. */
public enum class ScalePlacement {
    /** No scale in the bytes — the codes *are* the values, or the scale lives outside the buffer. */
    NONE,

    /** One or more scales at the start of each block (Q4_0's `d`, Q4_K's `d`/`dmin`). */
    BLOCK_HEAD,

    /** One or more scales at the end of each block (TQ1_0, TQ2_0, Q6_K). */
    BLOCK_TAIL,

    /** A single scale for the whole tensor, after the packed codes (BitNet b1.58). */
    PER_TENSOR,
}

/**
 * The block geometry of a packed [TensorEncoding] — the descriptor a decoder, a planner and a
 * fixture generator can all be driven from, so they cannot drift apart (SKEEP-003 §5.3; #988).
 *
 * `blockSize` elements occupy `bytesPerBlock` bytes. An encoding whose whole tensor is one block
 * (SKaiNET's own ternary packing, BitNet b1.58) reports [PER_TENSOR_BLOCK] and a `bytesPerBlock`
 * of `0`: for those, ask [TensorEncoding.physicalBytes] for the byte count and use
 * [bitsPerElement] for the rate.
 *
 * @property bitsPerElement payload bits per element, *excluding* per-block scales — the number
 *   that names the format ("1.58-bit", "4-bit"), not its on-disk rate; [amortisedBitsPerElement]
 *   is the honest one.
 * @property activation the format a kernel is expected to quantize the *other* operand to
 *   (`W1.58A8` → `Format(Int8, Dense(1))`), or `null` when the kernel consumes floats.
 */
public data class BlockSpec(
    val blockSize: Int,
    val bytesPerBlock: Int,
    val bitsPerElement: Double,
    val scale: ScalePlacement,
    val activation: Format? = null,
) {
    /** True when the tensor is a single block rather than a sequence of fixed-size ones. */
    public val isPerTensor: Boolean get() = blockSize == PER_TENSOR_BLOCK

    /**
     * Bits per element actually written, scales included — `bytesPerBlock * 8 / blockSize`
     * (2.06 for TQ2_0, 1.6875 for TQ1_0), or [bitsPerElement] for a per-tensor encoding.
     */
    public val amortisedBitsPerElement: Double
        get() = if (isPerTensor) bitsPerElement else bytesPerBlock * 8.0 / blockSize

    public companion object {
        /** [blockSize] value meaning "the whole tensor is one block". */
        public const val PER_TENSOR_BLOCK: Int = 0

        /**
         * The activation format of the ternary kernels (`W1.58A8`): int8 codes with a per-token
         * absmax scale — [TensorEncoding.DENSE_I8_ABSMAX], which is what the requant adapter of
         * #1040 produces and what `bitnet_gemv` consumes.
         */
        public val INT8_ACTIVATION: Format = Format(Int8, TensorEncoding.DENSE_I8_ABSMAX)
    }
}

/**
 * The [BlockSpec] of this encoding, or `null` when it is not block-structured ([TensorEncoding.Dense])
 * or its layout is unknown ([TensorEncoding.Opaque]).
 *
 * This table is the single source for the block geometry: the reference decoders
 * ([TernaryCodec]), the fixture generators and `EncodingSpecTest` all read it, so a wrong constant
 * fails a test rather than silently mis-decoding a file.
 */
public val TensorEncoding.blockSpec: BlockSpec?
    get() = when (this) {
        is TensorEncoding.Dense -> null
        is TensorEncoding.Opaque -> null
        // Activations, not weights: the "block" is a row, whose length is the tensor's, not the
        // encoding's — see the note on DENSE_I8_ABSMAX.
        TensorEncoding.DENSE_I8_ABSMAX -> null
        TensorEncoding.Q4_0 -> BlockSpec(32, 18, 4.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q5_0 -> BlockSpec(32, 22, 5.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q5_1 -> BlockSpec(32, 24, 5.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q8_0 -> BlockSpec(32, 34, 8.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q4_K -> BlockSpec(256, 144, 4.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q5_K -> BlockSpec(256, 176, 5.0, ScalePlacement.BLOCK_HEAD)
        TensorEncoding.Q6_K -> BlockSpec(256, 210, 6.0, ScalePlacement.BLOCK_TAIL)
        TensorEncoding.TQ1_0 -> BlockSpec(256, 54, 1.625, ScalePlacement.BLOCK_TAIL, BlockSpec.INT8_ACTIVATION)
        TensorEncoding.TQ2_0 -> BlockSpec(256, 66, 2.0, ScalePlacement.BLOCK_TAIL, BlockSpec.INT8_ACTIVATION)
        TensorEncoding.TernaryPacked ->
            BlockSpec(BlockSpec.PER_TENSOR_BLOCK, 0, 2.0, ScalePlacement.NONE, BlockSpec.INT8_ACTIVATION)
        TensorEncoding.BITNET_B1_58 ->
            BlockSpec(BlockSpec.PER_TENSOR_BLOCK, 0, 2.0, ScalePlacement.PER_TENSOR, BlockSpec.INT8_ACTIVATION)
        // Deliberately no BlockSpec and — unlike the other ternary encodings — NO int8 activation
        // hint: BITNET_PLANES is the f32-activation lm_head format (#1150); its geometry is
        // row-scoped ([rows] FP16 scales after 8 plane payloads), which the row-count-free
        // BlockSpec model cannot express, and without the planes kernel pack the dispatcher must
        // fall to the decoding reference matmul, never the int8 requantize adapter.
        TensorEncoding.BITNET_PLANES -> null
        is TensorEncoding.TurboQuantPolar ->
            BlockSpec(blockSize, (physicalBytes(blockSize.toLong()) ?: 0L).toInt(), bitsPerElement.toDouble(), ScalePlacement.BLOCK_HEAD)
        is TensorEncoding.TurboQuantPolarQjl ->
            BlockSpec(blockSize, (physicalBytes(blockSize.toLong()) ?: 0L).toInt(), (bitsPerElement + residualBits).toDouble(), ScalePlacement.BLOCK_HEAD)
    }

/** True when this encoding stores ternary values (`-1, 0, +1`) — the M2 kernel family. */
public val TensorEncoding.isTernary: Boolean
    get() = this == TensorEncoding.TQ1_0 || this == TensorEncoding.TQ2_0 ||
        this == TensorEncoding.TernaryPacked || this == TensorEncoding.BITNET_B1_58
