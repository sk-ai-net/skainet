package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data for the GGML **Q5_1** quantized format (5-bit, per-block minimum).
 *
 * Block format (32 elements, 24 bytes/block):
 * - bytes 0..1  : `d`  (f16 scale)
 * - bytes 2..3  : `m`  (f16 minimum)
 * - bytes 4..7  : `qh[0..3]` (5th/high bit of each of the 32 codes)
 * - bytes 8..23 : `qs[0..15]` (low 4 bits, two nibbles per byte)
 *
 * Dequant (matches `sk.ainet.io.gguf.dequant.DequantOps.dequantQ5_1FromBytes`),
 * for `j ∈ [0,16)`, `lo = qs[j] & 0x0F`, `hi = qs[j] >>> 4`,
 * `bitLo = (qh[j/8] >>> (j%8)) & 1`, `bitHi = (qh[(j+16)/8] >>> ((j+16)%8)) & 1`:
 *
 *   element[j]      = d * (lo + (bitLo shl 4)) + m
 *   element[j + 16] = d * (hi + (bitHi shl 4)) + m
 *
 * Block order: **canonical row-major** (`o * blocksPerRow + b`) — what a GGUF holds
 * and what this type's [dequantizeBlock] and `toFloatArray` assume. `Q5_1MatmulKernel`
 * reads *input-block-major* bytes instead, so a weight reaches it through a
 * relayout (`TensorView.prepack`), never by reinterpreting these bytes in place.
 * The contract is written down once in `the Packed weight layout page in the docs site (explanation/packed-weight-layout)`
 * (#973); this kdoc used to claim the opposite, which is the confusion that issue
 * exists to end.
 */
public interface Q5_1TensorData : TensorData<DType, Byte> {
    public val blockCount: Int
    public val packedData: ByteArray

    public companion object {
        public const val BLOCK_SIZE: Int = 32
        public const val BYTES_PER_BLOCK: Int = 24
    }
}

/** Packed-byte implementation of [Q5_1TensorData]. */
public class Q5_1BlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q5_1TensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data
    override val blockCount: Int = (shape.volume + Q5_1TensorData.BLOCK_SIZE - 1) / Q5_1TensorData.BLOCK_SIZE
    override val encoding: TensorEncoding get() = TensorEncoding.Q5_1
    override val blockSize: Int get() = Q5_1TensorData.BLOCK_SIZE

    init {
        val required = blockCount * Q5_1TensorData.BYTES_PER_BLOCK
        require(data.size >= required) { "Data size ${data.size} < required $required for $blockCount blocks" }
    }

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        val base = blockIdx * Q5_1TensorData.BYTES_PER_BLOCK
        val d = Q4_0BlockTensorData.halfToFloat(((data[base + 1].toInt() and 0xFF) shl 8) or (data[base].toInt() and 0xFF))
        val m = Q4_0BlockTensorData.halfToFloat(((data[base + 3].toInt() and 0xFF) shl 8) or (data[base + 2].toInt() and 0xFF))
        val qh = intArrayOf(
            data[base + 4].toInt() and 0xFF, data[base + 5].toInt() and 0xFF,
            data[base + 6].toInt() and 0xFF, data[base + 7].toInt() and 0xFF,
        )
        val qs = base + 8
        val elems = minOf(Q5_1TensorData.BLOCK_SIZE, shape.volume - blockIdx * Q5_1TensorData.BLOCK_SIZE)
        for (j in 0 until 16) {
            val q = data[qs + j].toInt() and 0xFF
            val lo = q and 0x0F; val hi = q ushr 4
            val bitLo = (qh[j / 8] ushr (j % 8)) and 1
            val bitHi = (qh[(j + 16) / 8] ushr ((j + 16) % 8)) and 1
            if (j < elems) output[outputOffset + j] = d * (lo + (bitLo shl 4)) + m
            if (16 + j < elems) output[outputOffset + 16 + j] = d * (hi + (bitHi shl 4)) + m
        }
    }

    override fun get(vararg indices: Int): Byte {
        val flat = calcFlatIndex(indices)
        val tmp = FloatArray(Q5_1TensorData.BLOCK_SIZE)
        dequantizeBlock(flat / Q5_1TensorData.BLOCK_SIZE, tmp, 0)
        return tmp[flat % Q5_1TensorData.BLOCK_SIZE].toInt().toByte()
    }

    override fun set(vararg indices: Int, value: Byte): Unit =
        throw UnsupportedOperationException("Q5_1BlockTensorData is read-only")

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match dimensions (${shape.dimensions.size})"
        }
        var flat = 0
        for (i in indices.indices) flat += indices[i] * strides[i]
        return flat
    }

    public companion object {
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q5_1BlockTensorData = Q5_1BlockTensorData(shape, bytes)
    }
}
