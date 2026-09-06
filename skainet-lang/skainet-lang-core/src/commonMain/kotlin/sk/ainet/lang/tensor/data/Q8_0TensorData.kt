package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for Q8_0 quantized format.
 *
 * Q8_0 block format (32 elements per block, 34 bytes per block):
 * - 2 bytes: f16 scale
 * - 32 bytes: int8 quantized codes
 *
 * Dequantization: output[i] = code[i] * scale
 *
 * This interface enables direct quantized matmul operations without full dequantization,
 * providing significant memory and compute savings for inference.
 */
public interface Q8_0TensorData : TensorData<DType, Byte> {
    /** Number of Q8_0 blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    /** Get the scale factor for a specific block. */
    public fun getBlockScale(blockIdx: Int): Float

    /** Get a quantized code value within a block (0..31). */
    public fun getCode(blockIdx: Int, elementIdx: Int): Byte

    public companion object {
        /** Elements per Q8_0 block. */
        public const val BLOCK_SIZE: Int = 32

        /** Bytes per Q8_0 block (2 bytes scale + 32 bytes codes). */
        public const val BYTES_PER_BLOCK: Int = 34
    }
}

/**
 * Implementation of Q8_0TensorData backed by a packed byte array.
 *
 * Memory layout per block:
 * - bytes [0..1]: f16 scale (little-endian)
 * - bytes [2..33]: 32 int8 quantized codes
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param packedData the raw packed block data
 */
public class Q8_0BlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q8_0TensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q8_0TensorData.BLOCK_SIZE - 1) / Q8_0TensorData.BLOCK_SIZE

    // PackedBlockStorage implementation
    override val encoding: TensorEncoding get() = TensorEncoding.Q8_0
    override val blockSize: Int get() = Q8_0TensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val scale = getBlockScale(blockIdx)
        val elemsInBlock = minOf(Q8_0TensorData.BLOCK_SIZE, shape.volume - blockIdx * Q8_0TensorData.BLOCK_SIZE)
        for (i in 0 until elemsInBlock) {
            val outIdx = outputOffset + i
            if (outIdx >= output.size) return
            output[outIdx] = getCode(blockIdx, i).toFloat() * scale
        }
    }

    init {
        val requiredBytes = blockCount * Q8_0TensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockScale(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val offset = blockIdx * Q8_0TensorData.BYTES_PER_BLOCK
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getCode(blockIdx: Int, elementIdx: Int): Byte {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(elementIdx in 0 until Q8_0TensorData.BLOCK_SIZE) { "Element index $elementIdx out of bounds (0..31)" }
        val offset = blockIdx * Q8_0TensorData.BYTES_PER_BLOCK + 2 + elementIdx
        return data[offset]
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q8_0TensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q8_0TensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx)
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q8_0TensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q8_0TensorData.BLOCK_SIZE
        val offset = blockIdx * Q8_0TensorData.BYTES_PER_BLOCK + 2 + elementIdx
        data[offset] = value
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
        /**
         * Create Q8_0TensorData from raw GGUF bytes.
         * Validates that the byte array has the expected size.
         */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q8_0BlockTensorData {
            return Q8_0BlockTensorData(shape, bytes)
        }

        /**
         * Convert f16 bits to float32.
         */
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
                        val floatExp = (e + 127) shl 23
                        val floatMant = m shl 13
                        Float.fromBits(sign or floatExp or floatMant)
                    }
                }
                31 -> {
                    val floatExp = 0xFF shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
                else -> {
                    val floatExp = (exp - 15 + 127) shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
            }
        }
    }
}

/**
 * Dequantize Q8_0 tensor data to a FloatArray.
 * output[i] = code[i] * scale
 */
public fun Q8_0TensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    var outIdx = 0
    for (blockIdx in 0 until blockCount) {
        val scale = getBlockScale(blockIdx)
        val elemsInBlock = minOf(Q8_0TensorData.BLOCK_SIZE, shape.volume - outIdx)
        for (i in 0 until elemsInBlock) {
            result[outIdx++] = getCode(blockIdx, i).toFloat() * scale
        }
    }
    return result
}
