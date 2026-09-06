package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.Ternary

/**
 * Marker interface for tensor data containing ternary values {-1, 0, +1}.
 *
 * This interface enables specialized ternary-aware operations in TensorOps,
 * such as addition-only matrix multiplication that avoids FP multiplies.
 *
 * Implementations may store data in various compact formats (2-bit, base-3, etc.)
 * but must provide element access returning Byte values of -1, 0, or +1.
 */
public interface TernaryTensorData : TensorData<Ternary, Byte> {
    /**
     * Scale factor for dequantization to FP32.
     * output[i] = ternaryValue[i] * scale
     */
    public val scale: Float

    /**
     * Access to the underlying packed byte array for efficient kernel operations.
     * The packing format is implementation-defined.
     */
    public val packedData: ByteArray
}

/**
 * Ternary tensor data using 2-bit encoding compatible with TQ2_0 format.
 *
 * Encoding scheme (matches GGUF TQ2_0):
 * - 0 → -1
 * - 1 → 0
 * - 2 → +1
 * - 3 → (reserved, treated as +1 for safety)
 *
 * This is the simpler ternary format with 4 values per byte.
 *
 * @param initialShape the shape of the tensor
 * @param data the packed byte array (4 ternary values per byte)
 * @param scale the scale factor for FP32 dequantization
 */
public class Ternary2BitTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    override val scale: Float = 1.0f
) : TernaryTensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    // PackedBlockStorage — treat the whole tensor as a single block
    override val encoding: TensorEncoding get() = TensorEncoding.TernaryPacked
    override val blockSize: Int get() = shape.volume
    override val blockCount: Int get() = 1

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx == 0) { "Ternary has a single block, got index $blockIdx" }
        for (i in 0 until shape.volume) {
            val byteIndex = i / 4
            val bitOffset = (i % 4) * 2
            val encoded = (data[byteIndex].toInt() ushr bitOffset) and 0x03
            val ternary = encoded - 1 // decode: 0→-1, 1→0, 2→+1
            val outIdx = outputOffset + i
            if (outIdx >= output.size) return
            output[outIdx] = ternary.toFloat() * scale
        }
    }

    init {
        val requiredBytes = (shape.volume + 3) / 4  // 4 values per byte
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for ${shape.volume} elements"
        }
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val byteIndex = flatIndex / 4
        val bitOffset = (flatIndex % 4) * 2

        val encoded = (data[byteIndex].toInt() ushr bitOffset) and 0x03
        return decodeToTernary(encoded)
    }

    override fun set(vararg indices: Int, value: Byte) {
        require(value in -1..1) { "Ternary value must be -1, 0, or 1, got $value" }

        val flatIndex = calcFlatIndex(indices)
        val byteIndex = flatIndex / 4
        val bitOffset = (flatIndex % 4) * 2

        val encoded = encodeFromTernary(value.toInt())

        // Clear existing 2 bits and set new value
        val mask = (0x03 shl bitOffset).inv()
        data[byteIndex] = ((data[byteIndex].toInt() and mask) or (encoded shl bitOffset)).toByte()
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
        /** Decode TQ2_0 encoded value {0,1,2} to ternary {-1,0,+1} */
        private fun decodeToTernary(encoded: Int): Byte = (encoded - 1).toByte()

        /** Encode ternary {-1,0,+1} to TQ2_0 format {0,1,2} */
        private fun encodeFromTernary(ternary: Int): Int = ternary + 1

        /**
         * Create from a TQ2_0 formatted block.
         * TQ2_0 block: 64 bytes data + 2 bytes f16 scale = 66 bytes total for 256 elements.
         */
        public fun fromTQ2_0Block(blockData: ByteArray, shape: Shape): Ternary2BitTensorData {
            require(blockData.size >= 66) { "TQ2_0 block must be at least 66 bytes" }

            val scaleBits = (blockData[65].toInt() and 0xFF shl 8) or (blockData[64].toInt() and 0xFF)
            val scale = halfToFloat(scaleBits)

            // TQ2_0 interleaves: byte `j + m` holds four elements 32 apart, so the bytes cannot be
            // adopted verbatim — decode them through the reference codec (#1033) and re-pack into
            // this type's four-consecutive-elements-per-byte layout.
            val codes = sk.ainet.lang.memory.TernaryCodec.codesTq2_0(blockData, minOf(shape.volume, 256))
            return fromTernaryValues(shape, codes.copyOf(shape.volume), scale)
        }

        /**
         * Create a new tensor with all zeros (ternary 0 encoded as 1).
         */
        public fun zeros(shape: Shape): Ternary2BitTensorData {
            val requiredBytes = (shape.volume + 3) / 4
            // Fill with 0x55 = 01 01 01 01 = all zeros in ternary
            val data = ByteArray(requiredBytes) { 0x55 }
            return Ternary2BitTensorData(shape, data)
        }

        /**
         * Create from explicit ternary values (-1, 0, or +1).
         */
        public fun fromTernaryValues(shape: Shape, values: ByteArray, scale: Float = 1.0f): Ternary2BitTensorData {
            require(values.size == shape.volume) {
                "Values size ${values.size} must match shape volume ${shape.volume}"
            }

            val requiredBytes = (shape.volume + 3) / 4
            val packed = ByteArray(requiredBytes)

            for (i in values.indices) {
                val byteIndex = i / 4
                val bitOffset = (i % 4) * 2
                val encoded = encodeFromTernary(values[i].toInt())
                packed[byteIndex] = (packed[byteIndex].toInt() or (encoded shl bitOffset)).toByte()
            }

            return Ternary2BitTensorData(shape, packed, scale)
        }

        private fun halfToFloat(hbits: Int): Float {
            val sign = (hbits and 0x8000) shl 16
            val exp = (hbits and 0x7C00) shr 10
            val mant = hbits and 0x03FF

            return when (exp) {
                0 -> {
                    // Subnormal or zero
                    if (mant == 0) {
                        Float.fromBits(sign)  // ±0
                    } else {
                        // Subnormal: convert to normalized float
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
                    // Infinity or NaN
                    val floatExp = 0xFF shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
                else -> {
                    // Normal number: convert exponent and mantissa
                    val floatExp = (exp - 15 + 127) shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
            }
        }
    }
}

/**
 * Convert a ternary tensor to FP32 by applying the scale factor.
 * output[i] = ternaryValue[i] * scale
 */
public fun TernaryTensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    val dims = shape.dimensions

    when (dims.size) {
        1 -> {
            for (i in 0 until dims[0]) {
                result[i] = get(i).toFloat() * scale
            }
        }
        2 -> {
            var idx = 0
            for (i in 0 until dims[0]) {
                for (j in 0 until dims[1]) {
                    result[idx++] = get(i, j).toFloat() * scale
                }
            }
        }
        else -> {
            // Generic multi-dimensional iteration
            val indices = IntArray(dims.size)
            for (i in 0 until shape.volume) {
                result[i] = get(*indices).toFloat() * scale

                // Increment indices
                for (d in dims.size - 1 downTo 0) {
                    indices[d]++
                    if (indices[d] < dims[d]) break
                    indices[d] = 0
                }
            }
        }
    }

    return result
}
