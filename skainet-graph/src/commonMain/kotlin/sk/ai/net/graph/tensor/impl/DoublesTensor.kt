package sk.ai.net.graph.tensor.impl

import sk.ai.net.core.TypedTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.core.Slice
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A tensor implementation that stores double values.
 *
 * This class implements Tensor and TypedTensor<Double> to provide
 * a tensor that stores double values and supports slicing.
 *
 * @property shape The shape of the tensor.
 * @property data The flat array containing the tensor's data.
 */
class DoublesTensor(
    override val shape: Shape,
    private val data: DoubleArray
) : Tensor, TypedTensor<Double> {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param data The flat array containing the tensor's data.
     */
    constructor(dimensions: List<Int>, data: DoubleArray) : this(Shape(dimensions.toIntArray()), data)

    /**
     * Retrieves the value at the specified indices.
     *
     * The indices are converted to a flat index using the tensor's shape, and the
     * corresponding value is retrieved from the flat data array.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    override fun get(vararg indices: Int): Double {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }
        return data[flatIndex]
    }

    /**
     * Retrieves a slice of the tensor.
     *
     * @param slices The slices to apply to the tensor.
     * @return A new tensor containing the sliced data.
     */
    override operator fun get(vararg slices: Slice): TypedTensor<Double> {
        // Calculate the new shape based on the slices
        val newDimensions = IntArray(shape.dimensions.size)
        for (i in shape.dimensions.indices) {
            if (i < slices.size) {
                val slice = slices[i]
                newDimensions[i] = (slice.endIndex - slice.startIndex).toInt()
            } else {
                newDimensions[i] = shape.dimensions[i]
            }
        }

        // Create a new tensor with the new shape
        val newShape = Shape(newDimensions)
        val newData = DoubleArray(newShape.dimensions.fold(1) { acc, dim -> acc * dim })

        // Copy the data from the original tensor to the new tensor
        // This is a simplified implementation that only works for 2D tensors
        if (shape.dimensions.size == 2 && slices.size == 2) {
            var newIndex = 0
            for (i in slices[0].startIndex until slices[0].endIndex) {
                for (j in slices[1].startIndex until slices[1].endIndex) {
                    val oldIndex = (i * shape.dimensions[1] + j).toInt()
                    newData[newIndex++] = data[oldIndex]
                }
            }
        }

        return DoublesTensor(newShape, newData)
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape and data.
     */
    override fun toString() =
        "DoublesTensor(shape=$shape, data=${data.contentToString()})"

    /**
     * Raises each element of the tensor to the power of the given scalar.
     *
     * This method creates a new tensor where each element is the result of
     * raising the corresponding element in the original tensor to the power
     * of the provided scalar value.
     *
     * @param scalar The exponent to raise each element to.
     * @return A new tensor with each element raised to the power of the scalar.
     */
    fun pow(scalar: Double): TypedTensor<Double> {
        val newData = DoubleArray(data.size) { i -> Math.pow(data[i], scalar) }
        return DoublesTensor(shape, newData)
    }
}

/**
 * Creates a tensor with the given shape and values.
 *
 * @param shape The shape of the tensor.
 * @param values The values to store in the tensor.
 * @return A new tensor with the given shape and values.
 */
fun createTensor(shape: Shape, values: DoubleArray): TypedTensor<Double> {
    return DoublesTensor(shape, values)
}

/**
 * Creates a tensor with the given shape and values.
 *
 * @param shape The shape of the tensor.
 * @param values The values to store in the tensor as an IntArray.
 * @return A new tensor with the given shape and values.
 */
fun createTensor(shape: Shape, values: IntArray): TypedTensor<Double> {
    return DoublesTensor(shape, values.map { it.toDouble() }.toDoubleArray())
}
