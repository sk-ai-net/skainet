package sk.ai.net.graph.autodiff

import sk.ai.net.graph.core.ComputeNode

/**
 * A computation node that supports automatic differentiation.
 *
 * This class extends [ComputeNode] and implements [Differentiable], providing
 * the foundation for automatic differentiation in computation graphs.
 *
 * @param T The type of data processed by this computation node.
 */
abstract class DifferentiableComputeNode<T> : ComputeNode<T>(), Differentiable<T> {
    /**
     * Computes the gradient of the output with respect to all inputs.
     *
     * @param outputGradient The gradient of the output with respect to this node.
     * @return A list of gradients, one for each input.
     */
    fun backpropagate(outputGradient: T): List<T> {
        return inputs.indices.map { gradient(it, outputGradient) }
    }
}