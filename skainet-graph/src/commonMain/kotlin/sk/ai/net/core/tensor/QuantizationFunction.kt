package sk.ai.net.core.tensor

/**
 * Interface for quantization functions that convert floating-point values to quantized values.
 *
 * This interface defines the contract for quantization functions that can be used
 * with quantized tensor implementations like Int8Tensor and Int4Tensor.
 */
interface QuantizationFunction {
    /**
     * Quantizes a DoubleArray to a ByteArray using the quantization parameters.
     *
     * @param values The double values to quantize.
     * @return A QuantizationResult containing the quantized data, scale, and zero point.
     */
    fun quantize(values: DoubleArray): QuantizationResult
}

/**
 * Data class representing the result of a quantization operation.
 *
 * @property data The quantized data as a ByteArray.
 * @property scale The scale factor used for quantization.
 * @property zeroPoint The zero point used for quantization.
 */
data class QuantizationResult(
    val data: ByteArray,
    val scale: Double,
    val zeroPoint: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as QuantizationResult

        if (!data.contentEquals(other.data)) return false
        if (scale != other.scale) return false
        if (zeroPoint != other.zeroPoint) return false

        return true
    }

    override fun hashCode(): Int {
        var result = data.contentHashCode()
        result = 31 * result + scale.hashCode()
        result = 31 * result + zeroPoint
        return result
    }
}

/**
 * Default linear quantization function for Int8 tensors.
 *
 * This function quantizes values to the range [-128, 127] using linear quantization.
 */
class LinearInt8QuantizationFunction : QuantizationFunction {
    override fun quantize(values: DoubleArray): QuantizationResult {
        // Find min and max values
        var minValue = Double.MAX_VALUE
        var maxValue = Double.MIN_VALUE

        for (value in values) {
            if (value < minValue) minValue = value
            if (value > maxValue) maxValue = value
        }

        // Calculate scale and zero point for quantization
        val range = maxValue - minValue
        val scale = if (range > 0) range / 255.0 else 1.0
        val zeroPoint = if (range > 0) ((-minValue / scale).toInt().coerceIn(-128, 127)) else 0

        // Create the byte array and quantize values
        val data = ByteArray(values.size)

        for (i in values.indices) {
            val quantizedValue = (values[i] / scale + zeroPoint).toInt().coerceIn(-128, 127)
            data[i] = quantizedValue.toByte()
        }

        return QuantizationResult(data, scale, zeroPoint)
    }
}

/**
 * Symmetric quantization function for Int8 tensors.
 *
 * This function quantizes values to the range [-127, 127] using symmetric quantization
 * (zero point is always 0).
 */
class SymmetricInt8QuantizationFunction : QuantizationFunction {
    override fun quantize(values: DoubleArray): QuantizationResult {
        // Find max absolute value
        var maxAbsValue = 0.0

        for (value in values) {
            val absValue = kotlin.math.abs(value)
            if (absValue > maxAbsValue) maxAbsValue = absValue
        }

        // Calculate scale for quantization (zero point is always 0 for symmetric quantization)
        val scale = if (maxAbsValue > 0) maxAbsValue / 127.0 else 1.0
        val zeroPoint = 0

        // Create the byte array and quantize values
        val data = ByteArray(values.size)

        for (i in values.indices) {
            val quantizedValue = (values[i] / scale).toInt().coerceIn(-127, 127)
            data[i] = quantizedValue.toByte()
        }

        return QuantizationResult(data, scale, zeroPoint)
    }
}

/**
 * Default linear quantization function for Int4 tensors.
 *
 * This function quantizes values to the range [-8, 7] using linear quantization.
 */
class LinearInt4QuantizationFunction : QuantizationFunction {
    override fun quantize(values: DoubleArray): QuantizationResult {
        // Find min and max values
        var minValue = Double.MAX_VALUE
        var maxValue = Double.MIN_VALUE

        for (value in values) {
            if (value < minValue) minValue = value
            if (value > maxValue) maxValue = value
        }

        // Calculate scale and zero point for quantization
        val range = maxValue - minValue
        val scale = if (range > 0) range / 15.0 else 1.0
        val zeroPoint = if (range > 0) ((-minValue / scale).toInt().coerceIn(-8, 7)) else 0

        // Create the byte array and quantize values
        val data = ByteArray(values.size)

        for (i in values.indices) {
            val quantizedValue = (values[i] / scale + zeroPoint).toInt().coerceIn(-8, 7)
            data[i] = quantizedValue.toByte()
        }

        // Print debug information for the first few values
        if (values.size > 0) {
            println("[DEBUG_LOG] LinearInt4QuantizationFunction: First value: ${values[0]}, quantized: ${data[0]}, scale: $scale, zeroPoint: $zeroPoint")
        }

        return QuantizationResult(data, scale, zeroPoint)
    }
}

/**
 * Symmetric quantization function for Int4 tensors.
 *
 * This function quantizes values to the range [-7, 7] using symmetric quantization
 * (zero point is always 0).
 */
class SymmetricInt4QuantizationFunction : QuantizationFunction {
    override fun quantize(values: DoubleArray): QuantizationResult {
        // Find max absolute value
        var maxAbsValue = 0.0

        for (value in values) {
            val absValue = kotlin.math.abs(value)
            if (absValue > maxAbsValue) maxAbsValue = absValue
        }

        // Calculate scale for quantization (zero point is always 0 for symmetric quantization)
        val scale = if (maxAbsValue > 0) maxAbsValue / 7.0 else 1.0
        val zeroPoint = 0

        // Create the byte array and quantize values
        val data = ByteArray(values.size)

        for (i in values.indices) {
            val quantizedValue = (values[i] / scale).toInt().coerceIn(-7, 7)
            data[i] = quantizedValue.toByte()
        }

        return QuantizationResult(data, scale, zeroPoint)
    }
}

/**
 * Factory for creating quantization functions.
 */
object QuantizationFunctions {
    /**
     * Creates a linear quantization function for Int8 tensors.
     */
    fun linearInt8(): QuantizationFunction = LinearInt8QuantizationFunction()

    /**
     * Creates a symmetric quantization function for Int8 tensors.
     */
    fun symmetricInt8(): QuantizationFunction = SymmetricInt8QuantizationFunction()

    /**
     * Creates a linear quantization function for Int4 tensors.
     */
    fun linearInt4(): QuantizationFunction = LinearInt4QuantizationFunction()

    /**
     * Creates a symmetric quantization function for Int4 tensors.
     */
    fun symmetricInt4(): QuantizationFunction = SymmetricInt4QuantizationFunction()

    /**
     * Creates a custom quantization function with the specified parameters.
     *
     * @param minValue The minimum value in the input range.
     * @param maxValue The maximum value in the input range.
     * @param bits The number of bits to use for quantization (4 or 8).
     * @param symmetric Whether to use symmetric quantization (zero point is always 0).
     */
    fun custom(minValue: Double, maxValue: Double, bits: Int, symmetric: Boolean = false): QuantizationFunction {
        return object : QuantizationFunction {
            override fun quantize(values: DoubleArray): QuantizationResult {
                val range = maxValue - minValue

                // Calculate quantization parameters based on bits and symmetry
                val (scale, zeroPoint) = when {
                    symmetric && bits == 8 -> {
                        val maxAbsValue = kotlin.math.max(kotlin.math.abs(minValue), kotlin.math.abs(maxValue))
                        Pair(maxAbsValue / 127.0, 0)
                    }
                    symmetric && bits == 4 -> {
                        val maxAbsValue = kotlin.math.max(kotlin.math.abs(minValue), kotlin.math.abs(maxValue))
                        Pair(maxAbsValue / 7.0, 0)
                    }
                    bits == 8 -> {
                        val scale = range / 255.0
                        val zeroPoint = ((-minValue / scale).toInt().coerceIn(-128, 127))
                        Pair(scale, zeroPoint)
                    }
                    bits == 4 -> {
                        val scale = range / 15.0
                        val zeroPoint = ((-minValue / scale).toInt().coerceIn(-8, 7))
                        Pair(scale, zeroPoint)
                    }
                    else -> throw IllegalArgumentException("Bits must be 4 or 8")
                }

                // Create the byte array and quantize values
                val data = ByteArray(values.size)

                for (i in values.indices) {
                    val quantizedValue = when {
                        symmetric && bits == 8 -> (values[i] / scale).toInt().coerceIn(-127, 127)
                        symmetric && bits == 4 -> (values[i] / scale).toInt().coerceIn(-7, 7)
                        bits == 8 -> (values[i] / scale + zeroPoint).toInt().coerceIn(-128, 127)
                        bits == 4 -> (values[i] / scale + zeroPoint).toInt().coerceIn(-8, 7)
                        else -> throw IllegalArgumentException("Bits must be 4 or 8")
                    }
                    data[i] = quantizedValue.toByte()
                }

                return QuantizationResult(data, scale, zeroPoint)
            }
        }
    }
}
