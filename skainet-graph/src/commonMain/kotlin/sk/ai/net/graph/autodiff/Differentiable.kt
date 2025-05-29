package sk.ai.net.graph.autodiff

import sk.ai.net.graph.core.ComputeNode

/**
 * Interface for differentiable computation nodes.
 *
 * A differentiable node is a computation node that can compute its gradient
 * with respect to its inputs. This is used for automatic differentiation,
 * which is a key component of gradient-based optimization algorithms like
 * those used in deep learning.
 *
 * @param T The type of data processed by the computation node.
 */
interface Differentiable<T> {
    /**
     * Computes the gradient of this node with respect to the specified input.
     *
     * @param inputIndex The index of the input to compute the gradient with respect to.
     * @param gradient The gradient of the output with respect to this node.
     * @return The gradient of this node with respect to the specified input.
     */
    fun gradient(inputIndex: Int, gradient: T): T
}