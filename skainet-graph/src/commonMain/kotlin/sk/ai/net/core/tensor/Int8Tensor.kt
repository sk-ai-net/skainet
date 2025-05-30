package sk.ai.net.core.tensor

import sk.ai.net.core.tensor.shape.Shape

/**
 * A memory-efficient implementation of the Tensor interface for 8-bit integer values.
 *
 * This class represents a multi-dimensional array of 8-bit integer values (-128 to 127)
 * stored in a byte array. Each value is represented using 8 bits (1 byte),
 * which is more memory-efficient than using 8 bytes per value (as in a DoubleArray).
 *
 * @property shape The shape of the tensor, represented as a Shape data class containing dimensions.
 * @property data The byte array containing the 8-bit integer values.
 * @property scale The scaling factor used for quantization.
 * @property zeroPoint The zero point used for quantization.
 */
class Int8Tensor(
    override val shape: Shape,
    private val data: ByteArray,
    private val scale: Double,
    private val zeroPoint: Int
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param data The byte array containing the 8-bit integer values.
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
        /**
         * Creates an Int8Tensor from a DoubleArray using linear quantization.
         *
         * @param shape The shape of the tensor.
         * @param values The double values to convert to 8-bit integer values.
         * @param quantizationFunction Optional custom quantization function. If not provided, a default linear quantization is used.
         * @return A new Int8Tensor.
         */
        fun fromDoubles(
            shape: Shape, 
            values: DoubleArray,
            quantizationFunction: QuantizationFunction? = null
        ): Int8Tensor {
            // Calculate the total number of elements
            val totalElements = shape.dimensions.fold(1) { acc, dim -> acc * dim }

            // If a custom quantization function is provided, use it
            if (quantizationFunction != null) {
                val result = quantizationFunction.quantize(values)
                return Int8Tensor(shape, result.data, result.scale, result.zeroPoint)
            }

            // Otherwise, use default symmetric quantization
            val defaultQuantizer = SymmetricInt8QuantizationFunction()
            val result = defaultQuantizer.quantize(values)
            return Int8Tensor(shape, result.data, result.scale, result.zeroPoint)
        }

        /**
         * Creates an Int8Tensor from a DoubleArray using linear quantization.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param values The double values to convert to 8-bit integer values.
         * @param quantizationFunction Optional custom quantization function. If not provided, a default linear quantization is used.
         * @return A new Int8Tensor.
         */
        fun fromDoubles(
            dimensions: List<Int>, 
            values: DoubleArray,
            quantizationFunction: QuantizationFunction? = null
        ): Int8Tensor {
            return fromDoubles(Shape(dimensions.toIntArray()), values, quantizationFunction)
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

        // Dequantize the value
        // Ensure the byte is correctly interpreted as a signed value
        val quantizedValue = data[flatIndex].toInt()
        return (quantizedValue - zeroPoint) * scale
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape.
     */
    override fun toString(): String {
        return "Int8Tensor(shape=$shape, scale=$scale, zeroPoint=$zeroPoint)"
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
        return data[flatIndex]
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
