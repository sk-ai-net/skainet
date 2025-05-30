package sk.ai.net.core.backend

import sk.ai.net.core.tensor.Tensor

/**
 * Interface representing a computation backend for tensor operations.
 *
 * A computation backend is responsible for executing tensor operations on a specific
 * hardware platform (CPU, GPU, etc.). Different backends can provide different
 * implementations of the same operations, optimized for their target platform.
 */
interface ComputeBackend {
    /**
     * The name of the backend.
     */
    val name: String

    /**
     * Adds two tensors element-wise.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of adding the two tensors.
     */
    fun add(left: Tensor, right: Tensor): Tensor

    /**
     * Multiplies two tensors element-wise.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of multiplying the two tensors.
     */
    fun multiply(left: Tensor, right: Tensor): Tensor

    /**
     * Performs matrix multiplication of two tensors.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of matrix multiplication.
     */
    fun matmul(left: Tensor, right: Tensor): Tensor

    /**
     * Applies the ReLU activation function to a tensor.
     *
     * @param tensor The input tensor.
     * @return The result of applying ReLU.
     */
    fun relu(tensor: Tensor): Tensor
}