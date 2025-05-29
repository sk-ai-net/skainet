package sk.ai.net.graph.tensor

import sk.ai.net.graph.tensor.shape.Shape

/**
 * Interface representing a multi-dimensional array of numeric values.
 * 
 * A Tensor is a generalization of vectors and matrices to potentially higher dimensions.
 * The [shape] property defines the dimensions of the tensor, and the [get] operator
 * allows accessing individual elements by their indices.
 */
interface Tensor {
    /**
     * The shape of the tensor, represented as a Shape data class containing dimensions.
     * 
     * For example, a scalar has an empty shape, a vector has a shape with one dimension,
     * a matrix has a shape with two dimensions, and so on.
     */
    val shape: Shape

    /**
     * Retrieves the value at the specified indices.
     * 
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    operator fun get(vararg indices: Int): Double
}
