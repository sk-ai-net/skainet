@file:Suppress("DEPRECATION")

package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape

/**
 * Shared contract for all packed/quantized block tensor storage formats.
 *
 * Instead of each quantization format (Q4_K, Q8_0, Ternary, …) inventing
 * its own loader, planner, and backend handling path, all packed formats
 * implement this interface. Backends and planners can dispatch on
 * [encoding] without knowing every possible quantization scheme.
 *
 * Individual formats still expose format-specific accessors (sub-block
 * scales, code extraction, etc.) through their own sub-interfaces.
 */
public interface PackedBlockStorage {

    /** The logical shape of the tensor (element count, not block count). */
    public val shape: Shape

    /** The physical encoding describing the block layout. */
    public val encoding: TensorEncoding

    /** Number of blocks in this storage. */
    public val blockCount: Int

    /** Number of logical elements per block. */
    public val blockSize: Int

    /** Raw packed byte data containing all blocks. */
    public val packedData: ByteArray

    /**
     * The [sk.ainet.lang.memory.Storage] backing [packedData] (#1202).
     *
     * Defaults to wrapping [packedData] in [sk.ainet.lang.memory.Storage.Heap] — the historical
     * behavior, unchanged for every implementer that doesn't override this. An implementer whose
     * bytes actually live off-heap or mapped (e.g. a large ternary weight, to stay off the ART
     * heap cap) overrides this to expose its real backing storage instead.
     */
    public val packedStorage: sk.ainet.lang.memory.Storage
        get() = sk.ainet.lang.memory.Storage.Heap.wrap(packedData, mutable = false)

    /**
     * The order [packedData]'s blocks are physically in (#1120, #1124).
     *
     * Canonical storage — every GGUF-shaped producer — is
     * [sk.ainet.lang.memory.BlockOrder.ROW_MAJOR]: block `(o, b)` at flat index `o * blocksPerRow + b`.
     * A weight that has been relayouted for the packed matmul kernels is
     * [sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR] instead, and the two coincide only at one
     * block per row.
     *
     * Before this existed, a relayouted weight was wrapped in a data type that still claimed to be
     * canonical. The kernels that read [packedData] directly were unaffected — they were addressing
     * it in feed order deliberately — but anything decoding through [packedView] read the wrong
     * blocks and returned plausible garbage (#1124, and #973/#968 before it). Declaring the order is
     * what lets both readers be right about the same bytes.
     *
     * Defaults to `ROW_MAJOR`, so every existing implementation is unchanged.
     */
    public val blockOrder: sk.ainet.lang.memory.BlockOrder
        get() = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR

    /** Physical byte size of the packed data. */
    public val physicalBytes: Long get() = packedData.size.toLong()

    /** Logical element count. */
    public val elementCount: Long get() = shape.volume.toLong()

    /**
     * Dequantize a single block to float values.
     *
     * @param blockIdx  The block index (0-based)
     * @param output    Destination array (must have at least [blockSize] elements from [outputOffset])
     * @param outputOffset  Starting index in [output]
     */
    public fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int = 0)

    /**
     * Dequantize the entire tensor to a FloatArray.
     * Default implementation calls [dequantizeBlock] for each block.
     */
    /**
     * This packed data as a [sk.ainet.lang.memory.TensorView] — `Format(FP32, encoding)` over a
     * blocked [sk.ainet.lang.memory.Layout] whose storage **borrows** [packedData] (SKEEP-003 §4.1
     * façade, rule 5). Nothing is copied: slicing or transposing the view addresses whole blocks
     * and the bytes stay exactly as the loader produced them, which is what keeps every packed
     * kernel bit-identical.
     *
     * `view.get(...)` decodes through [dequantizeBlock] (rule 4) — it never returns a raw byte,
     * unlike this data's own `get`, which stays as it is for source compatibility.
     */
    public val packedView: sk.ainet.lang.memory.TensorView
        get() = sk.ainet.lang.memory.TensorView.packed(
            storage = packedStorage,
            shape = shape,
            encoding = encoding,
            decoder = sk.ainet.lang.memory.PackedBlockDecoder(this),
            blockOrder = blockOrder,
        )

    public fun toFloatArray(): FloatArray {
        val result = FloatArray(shape.volume)
        if (blockOrder == sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR && shape.rank == 2) {
            // Feed-order bytes hold block `(o, b)` at physical index `b * rows + o`, so decoding
            // them in physical sequence would emit the matrix transposed-in-blocks. Walk the
            // *logical* grid instead and fetch each block from where this order put it — which is
            // what makes a feed-order weight decode to the same matrix as the canonical one it was
            // permuted from (#1120).
            val rows = shape[0]
            val blocksPerRow = shape[1] / blockSize
            for (o in 0 until rows) {
                for (b in 0 until blocksPerRow) {
                    dequantizeBlock(b * rows + o, result, (o * blocksPerRow + b) * blockSize)
                }
            }
            return result
        }
        var offset = 0
        for (i in 0 until blockCount) {
            val remaining = shape.volume - offset
            dequantizeBlock(i, result, offset)
            offset += minOf(blockSize, remaining)
        }
        return result
    }

    /**
     * Convert this packed storage to a [TensorStorage] descriptor.
     */
    @Deprecated(message = "Use toTensorStorage(dtype, placement) (SKEEP-003 decision #13).", replaceWith = ReplaceWith("toTensorStorage(dtype, placement)"))
    public fun toTensorStorage(
        logicalType: LogicalDType = LogicalDType.FLOAT32,
        placement: Placement = Placement.CPU_HEAP
    ): TensorStorage = TensorStorage(
        shape = shape,
        logicalType = logicalType,
        encoding = encoding,
        buffer = BufferHandle.Borrowed(packedData, isMutable = false),
        placement = placement
    )

    /** Convert this packed storage to a [TensorStorage] descriptor, dtype-first (packed weights are logically [FP32]). */
    public fun toTensorStorage(
        dtype: sk.ainet.lang.types.DType,
        placement: Placement = Placement.CPU_HEAP
    ): TensorStorage = toTensorStorage(dtype.toLogicalDType(), placement)
}
