package sk.ai.net.graph.tensor

import sk.ai.net.graph.tensor.shape.Shape

/**
 * A memory-efficient implementation of the Tensor interface for 4-bit integer values.
 *
 * This class represents a multi-dimensional array of 4-bit integer values (-8 to 7)
 * stored in a bit-packed format. Each value is represented using 4 bits,
 * allowing 2 values per byte, which is significantly more memory-efficient than
 * using 8 bytes per value (as in a DoubleArray).
 *
 * @property shape The shape of the tensor, represented as a Shape data class containing dimensions.
 * @property data The byte array containing the bit-packed 4-bit integer values.
 * @property scale The scaling factor used for quantization.
 * @property zeroPoint The zero point used for quantization.
 */
class Int4Tensor(
    override val shape: Shape,
    private val data: ByteArray,
    private val scale: Double,
    private val zeroPoint: Int
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param data The byte array containing the bit-packed 4-bit integer values.
     * @param scale The scaling factor used for quantization.
     * @param zeroPoint The zero point used for quantization.
     */
    constructor(
        dimensions: List<Int>,
        data: ByteArray,
        scale: Double,
        zeroPoint: Int
    ) : this(Shape(dimensions.toIntArray()), data, scale, zeroPoint)
    companion object {
        // Number of values per byte (2 values, 4 bits each)
        private const val VALUES_PER_BYTE = 2

        // Bit masks for extracting values from a byte
        private val MASKS = arrayOf(
            0xF0.toByte(), // First value (bits 4-7)
            0x0F.toByte()  // Second value (bits 0-3)
        )

        // Shift amounts for extracting values from a byte
        private val SHIFTS = arrayOf(4, 0)

        /**
         * Creates an Int4Tensor from a DoubleArray using quantization.
         *
         * @param shape The shape of the tensor.
         * @param values The double values to convert to 4-bit integer values.
         * @param quantizationFunction Optional custom quantization function. If not provided, a default linear quantization is used.
         * @return A new Int4Tensor.
         */
        fun fromDoubles(
            shape: Shape,
            values: DoubleArray,
            quantizationFunction: QuantizationFunction? = null
        ): Int4Tensor {
            // Calculate the total number of elements
            val totalElements = shape.dimensions.fold(1) { acc, dim -> acc * dim }

            // If a custom quantization function is provided, use it
            if (quantizationFunction != null) {
                val result = quantizationFunction.quantize(values)

                // We need to pack the 8-bit values into 4-bit values
                val packedData = packBytes(result.data, totalElements)

                return Int4Tensor(shape, packedData, result.scale, result.zeroPoint)
            }

            // Otherwise, use default symmetric quantization
            val defaultQuantizer = SymmetricInt4QuantizationFunction()
            val result = defaultQuantizer.quantize(values)

            // We need to pack the 8-bit values into 4-bit values
            val packedData = packBytes(result.data, totalElements)

            return Int4Tensor(shape, packedData, result.scale, result.zeroPoint)
        }

        /**
         * Creates an Int4Tensor from a DoubleArray using quantization.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param values The double values to convert to 4-bit integer values.
         * @param quantizationFunction Optional custom quantization function. If not provided, a default linear quantization is used.
         * @return A new Int4Tensor.
         */
        fun fromDoubles(
            dimensions: List<Int>,
            values: DoubleArray,
            quantizationFunction: QuantizationFunction? = null
        ): Int4Tensor {
            return fromDoubles(Shape(dimensions.toIntArray()), values, quantizationFunction)
        }

        /**
         * Packs 8-bit values from a ByteArray into 4-bit values.
         *
         * @param bytes The ByteArray containing 8-bit values.
         * @param totalElements The total number of elements.
         * @return A new ByteArray with packed 4-bit values.
         */
        private fun packBytes(bytes: ByteArray, totalElements: Int): ByteArray {
            val packedSize = (totalElements + VALUES_PER_BYTE - 1) / VALUES_PER_BYTE
            val packedData = ByteArray(packedSize)

            // Print debug information for the first few values
            if (totalElements > 0) {
                println("[DEBUG_LOG] packBytes: First few values:")
                for (i in 0 until minOf(6, totalElements)) {
                    println("[DEBUG_LOG] bytes[$i] = ${bytes[i]}")
                }
            }

            for (i in 0 until totalElements) {
                // Get the 8-bit value and ensure it's in the 4-bit range [-8, 7]
                val value = bytes[i].toInt().coerceIn(-8, 7) and 0x0F

                val byteIndex = i / VALUES_PER_BYTE
                val valueIndex = i % VALUES_PER_BYTE
                val shift = SHIFTS[valueIndex]

                // Clear the bits for this value
                packedData[byteIndex] = (packedData[byteIndex].toInt() and MASKS[valueIndex].toInt().inv()).toByte()

                // Set the bits for this value
                packedData[byteIndex] = (packedData[byteIndex].toInt() or (value shl shift)).toByte()

                // Print debug information for the first few values
                if (i < 6) {
                    println("[DEBUG_LOG] packBytes: i=$i, bytes[i]=${bytes[i]}, value=$value, byteIndex=$byteIndex, valueIndex=$valueIndex, shift=$shift, packedData[byteIndex]=${packedData[byteIndex]}")
                }
            }

            return packedData
        }
    }

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The dequantized value at the specified indices.
     */
    override fun get(vararg indices: Int): Double {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }

        val byteIndex = flatIndex / VALUES_PER_BYTE
        val valueIndex = flatIndex % VALUES_PER_BYTE
        val mask = MASKS[valueIndex]
        val shift = SHIFTS[valueIndex]

        // Extract the 4-bit pattern
        val maskedValue = data[byteIndex].toInt() and mask.toInt()
        val pattern = (maskedValue and 0xFF) shr shift and 0x0F

        // Convert the pattern to a signed value (-8 to 7)
        val signedPattern = if (pattern > 7) pattern - 16 else pattern

        // Dequantize the value
        val result = (signedPattern - zeroPoint) * scale

        // Print debug information
        println("[DEBUG_LOG] Int4Tensor.get: indices=${indices.joinToString()}, flatIndex=$flatIndex, byteIndex=$byteIndex, valueIndex=$valueIndex, maskedValue=$maskedValue, pattern=$pattern, signedPattern=$signedPattern, zeroPoint=$zeroPoint, scale=$scale, result=$result")

        return result
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape.
     */
    override fun toString(): String {
        return "Int4Tensor(shape=$shape, scale=$scale, zeroPoint=$zeroPoint)"
    }

    /**
     * Calculates the memory usage of this tensor in bytes.
     *
     * @return The memory usage in bytes.
     */
    fun memoryUsage(): Int {
        // Size of the byte array + overhead for scale and zeroPoint
        return data.size + 12 // 8 bytes for Double scale + 4 bytes for Int zeroPoint
    }

    /**
     * Gets the raw quantized value at the specified indices without dequantization.
     *
     * @param indices The indices of the element to retrieve.
     * @return The raw quantized value at the specified indices.
     */
    fun getRawValue(vararg indices: Int): Byte {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }

        val byteIndex = flatIndex / VALUES_PER_BYTE
        val valueIndex = flatIndex % VALUES_PER_BYTE
        val mask = MASKS[valueIndex]
        val shift = SHIFTS[valueIndex]

        // Extract the 4-bit pattern
        val maskedValue = data[byteIndex].toInt() and mask.toInt()
        val pattern = (maskedValue and 0xFF) shr shift and 0x0F

        // Convert the pattern to a signed value (-8 to 7)
        val signedPattern = if (pattern > 7) pattern - 16 else pattern

        return signedPattern.toByte()
    }

    /**
     * Gets the scale factor used for quantization.
     *
     * @return The scale factor.
     */
    fun getScale(): Double = scale

    /**
     * Gets the zero point used for quantization.
     *
     * @return The zero point.
     */
    fun getZeroPoint(): Int = zeroPoint
}
