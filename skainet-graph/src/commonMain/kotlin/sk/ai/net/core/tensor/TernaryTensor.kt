package sk.ai.net.core.tensor

import sk.ai.net.core.tensor.shape.Shape

/**
 * A memory-efficient implementation of the Tensor interface for ternary values (-1, 0, 1).
 *
 * This class represents a multi-dimensional array of ternary values (-1, 0, 1) stored in a
 * bit-packed format. Each value is represented using 2 bits:
 * - 00 for 0
 * - 01 for 1
 * - 10 for -1
 *
 * This allows storing 4 values per byte, which is significantly more memory-efficient than
 * using 8 bytes per value (as in a DoubleArray).
 *
 * @property shape The shape of the tensor, represented as a Shape data class containing dimensions.
 * @property data The byte array containing the bit-packed ternary values.
 */
class TernaryTensor(
    override val shape: Shape,
    private val data: ByteArray
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param data The byte array containing the bit-packed ternary values.
     */
    constructor(
        dimensions: List<Int>,
        data: ByteArray
    ) : this(Shape(dimensions.toIntArray()), data)
    
    companion object {
        // Bit patterns for the ternary values
        private const val ZERO_PATTERN: Int = 0b00 // 0
        private const val ONE_PATTERN: Int = 0b01  // 1
        private const val NEG_ONE_PATTERN: Int = 0b10 // -1

        // Number of values per byte (4 values, 2 bits each)
        private const val VALUES_PER_BYTE = 4

        // Bit masks for extracting values from a byte
        private val MASKS = arrayOf(
            0b11000000.toByte(), // First value (bits 6-7)
            0b00110000.toByte(), // Second value (bits 4-5)
            0b00001100.toByte(), // Third value (bits 2-3)
            0b00000011.toByte()  // Fourth value (bits 0-1)
        )

        // Shift amounts for extracting values from a byte
        private val SHIFTS = arrayOf(6, 4, 2, 0)

        /**
         * Creates a TernaryTensor from a DoubleArray.
         *
         * @param shape The shape of the tensor.
         * @param values The double values to convert to ternary values.
         * @return A new TernaryTensor.
         */
        fun fromDoubles(shape: Shape, values: DoubleArray): TernaryTensor {
            // Calculate the total number of elements
            val totalElements = shape.dimensions.fold(1) { acc, dim -> acc * dim }

            // Calculate the required size of the byte array
            val byteSize = (totalElements + VALUES_PER_BYTE - 1) / VALUES_PER_BYTE

            // Create the byte array
            val data = ByteArray(byteSize)

            // Pack the values into the byte array
            for (i in 0 until totalElements) {
                val value = when {
                    values[i] > 0.5 -> ONE_PATTERN
                    values[i] < -0.5 -> NEG_ONE_PATTERN
                    else -> ZERO_PATTERN
                }

                val byteIndex = i / VALUES_PER_BYTE
                val valueIndex = i % VALUES_PER_BYTE
                val shift = SHIFTS[valueIndex]

                // Clear the bits for this value
                data[byteIndex] = (data[byteIndex].toInt() and MASKS[valueIndex].toInt().inv()).toByte()

                // Set the bits for this value
                data[byteIndex] = (data[byteIndex].toInt() or (value shl shift)).toByte()
            }

            return TernaryTensor(shape, data)
        }

        /**
         * Creates a TernaryTensor from a DoubleArray.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param values The double values to convert to ternary values.
         * @return A new TernaryTensor.
         */
        fun fromDoubles(dimensions: List<Int>, values: DoubleArray): TernaryTensor {
            return fromDoubles(Shape(dimensions.toIntArray()), values)
        }
    }

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices (-1.0, 0.0, or 1.0).
     */
    override fun get(vararg indices: Int): Double {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }

        val byteIndex = flatIndex / VALUES_PER_BYTE
        val valueIndex = flatIndex % VALUES_PER_BYTE
        val mask = MASKS[valueIndex]
        val shift = SHIFTS[valueIndex]

        // Extract the 2-bit pattern
        val maskedValue = data[byteIndex].toInt() and mask.toInt()
        // Use unsigned right shift (>>>) to avoid sign extension
        val pattern = (maskedValue and 0xFF) shr shift and 0b11

        // Convert the pattern to a double
        return when (pattern) {
            ONE_PATTERN -> 1.0
            NEG_ONE_PATTERN -> -1.0
            else -> 0.0
        }
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape.
     */
    override fun toString(): String {
        return "TernaryTensor(shape=$shape)"
    }

    /**
     * Calculates the memory usage of this tensor in bytes.
     *
     * @return The memory usage in bytes.
     */
    fun memoryUsage(): Int {
        // Size of the byte array + overhead
        return data.size
    }
}