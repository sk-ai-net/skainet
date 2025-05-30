package sk.ai.net.core.tensor

import sk.ai.net.core.tensor.shape.Shape

/**
 * A simple implementation of the Tensor interface.
 *
 * This class represents a multi-dimensional array of numeric values stored in a flat array.
 * The shape of the tensor defines how the flat array is interpreted as a multi-dimensional array.
 *
 * @property shape The shape of the tensor, represented as a Shape data class containing dimensions.
 * @property data The flat array containing the tensor's data.
 */
class SimpleTensor(
    override val shape: Shape,
    private val data: DoubleArray
) : Tensor {
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
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape and data.
     */
    override fun toString() =
        "Tensor(shape=$shape, data=${data.contentToString()})"
}
