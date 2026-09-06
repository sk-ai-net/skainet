package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for the Q4_0 quantized format (older GGML 4-bit).
 *
 * Q4_0 block format (32 elements per block, 18 bytes per block):
 * - 2 bytes: f16 scale (`d`)
 * - 16 bytes: 32 packed 4-bit codes (2 nibbles per byte)
 *
 * Canonical ggml nibble layout (the *split* layout, matching
 * `sk.ainet.io.gguf.dequant.DequantOps.dequantQ4_0FromBytes`): for the
 * 16 code bytes `qs[0..15]`, the low nibbles decode elements `0..15` and
 * the high nibbles decode elements `16..31`:
 *
 *   element[j]      = ((qs[j] & 0x0F) - 8) * d   for j ∈ [0, 16)
 *   element[j + 16] = ((qs[j] >>> 4) - 8) * d
 *
 * The `- 8` bias makes the 4-bit code symmetric around zero. This is the
 * layout real GGUF Q4_0 weights are stored in.
 *
 * This interface enables direct quantized matmul without full
 * dequantization, mirroring [Q8_0TensorData].
 */
public interface Q4_0TensorData : TensorData<DType, Byte> {
    /** Number of Q4_0 blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    /** Get the scale factor (`d`) for a specific block. */
    public fun getBlockScale(blockIdx: Int): Float

    /**
     * Get the raw unsigned 4-bit code (0..15) for [elementIdx] (0..31)
     * within a block. The dequantized value is `(code - 8) * scale`.
     */
    public fun getCode(blockIdx: Int, elementIdx: Int): Byte

    public companion object {
        /** Elements per Q4_0 block. */
        public const val BLOCK_SIZE: Int = 32

        /** Bytes per Q4_0 block (2 bytes scale + 16 bytes packed nibbles). */
        public const val BYTES_PER_BLOCK: Int = 18
    }
}

/**
 * Implementation of [Q4_0TensorData] backed by a packed byte array.
 *
 * Memory layout per block (18 bytes):
 * - bytes [0..1]  : f16 scale (little-endian)
 * - bytes [2..17] : 16 bytes packing 32 4-bit codes (split layout, see
 *   [Q4_0TensorData] kdoc)
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param data the raw packed block data
 */
public class Q4_0BlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q4_0TensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q4_0TensorData.BLOCK_SIZE - 1) / Q4_0TensorData.BLOCK_SIZE

    // PackedBlockStorage implementation
    override val encoding: TensorEncoding get() = TensorEncoding.Q4_0
    override val blockSize: Int get() = Q4_0TensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val scale = getBlockScale(blockIdx)
        val elemsInBlock = minOf(Q4_0TensorData.BLOCK_SIZE, shape.volume - blockIdx * Q4_0TensorData.BLOCK_SIZE)
        val codesBase = blockIdx * Q4_0TensorData.BYTES_PER_BLOCK + 2
        for (j in 0 until 16) {
            val b = data[codesBase + j].toInt() and 0xFF
            val lo = (b and 0x0F) - 8
            val hi = (b ushr 4) - 8
            val o0 = outputOffset + j
            if (j < elemsInBlock && o0 < output.size) output[o0] = lo.toFloat() * scale
            val o1 = outputOffset + 16 + j
            if (16 + j < elemsInBlock && o1 < output.size) output[o1] = hi.toFloat() * scale
        }
    }

    init {
        val requiredBytes = blockCount * Q4_0TensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockScale(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val offset = blockIdx * Q4_0TensorData.BYTES_PER_BLOCK
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        return halfToFloat((b1 shl 8) or b0)
    }

    override fun getCode(blockIdx: Int, elementIdx: Int): Byte {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(elementIdx in 0 until Q4_0TensorData.BLOCK_SIZE) { "Element index $elementIdx out of bounds (0..31)" }
        val byteInBlock = if (elementIdx < 16) elementIdx else elementIdx - 16
        val b = data[blockIdx * Q4_0TensorData.BYTES_PER_BLOCK + 2 + byteInBlock].toInt() and 0xFF
        val nibble = if (elementIdx < 16) (b and 0x0F) else (b ushr 4)
        return nibble.toByte()
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_0TensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_0TensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx)
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_0TensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_0TensorData.BLOCK_SIZE
        val byteInBlock = if (elementIdx < 16) elementIdx else elementIdx - 16
        val offset = blockIdx * Q4_0TensorData.BYTES_PER_BLOCK + 2 + byteInBlock
        val nib = value.toInt() and 0x0F
        val cur = data[offset].toInt() and 0xFF
        data[offset] = if (elementIdx < 16) ((cur and 0xF0) or nib).toByte()
        else ((cur and 0x0F) or (nib shl 4)).toByte()
    }

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }

    public companion object {
        /** Create [Q4_0BlockTensorData] from raw packed Q4_0 bytes. */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q4_0BlockTensorData {
            return Q4_0BlockTensorData(shape, bytes)
        }

        /** Convert f16 bits to float32. */
        internal fun halfToFloat(hbits: Int): Float {
            val sign = (hbits and 0x8000) shl 16
            val exp = (hbits and 0x7C00) shr 10
            val mant = hbits and 0x03FF
            return when (exp) {
                0 -> {
                    if (mant == 0) {
                        Float.fromBits(sign)
                    } else {
                        var m = mant
                        var e = -14
                        while ((m and 0x400) == 0) {
                            m = m shl 1
                            e--
                        }
                        m = m and 0x3FF
                        Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                    }
                }
                31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
                else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
            }
        }
    }
}

/**
 * Dequantize Q4_0 tensor data to a FloatArray.
 * `element[j] = (code[j] - 8) * scale` in the canonical split layout.
 */
public fun Q4_0TensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    for (blockIdx in 0 until blockCount) {
        val scale = getBlockScale(blockIdx)
        val base = blockIdx * Q4_0TensorData.BLOCK_SIZE
        val elemsInBlock = minOf(Q4_0TensorData.BLOCK_SIZE, shape.volume - base)
        for (i in 0 until elemsInBlock) {
            result[base + i] = (getCode(blockIdx, i).toInt() - 8).toFloat() * scale
        }
    }
    return result
}
