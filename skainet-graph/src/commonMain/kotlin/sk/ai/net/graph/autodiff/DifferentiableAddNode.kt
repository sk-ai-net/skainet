package sk.ai.net.graph.autodiff

import sk.ai.net.graph.core.AddNode

/**
 * A differentiable node that adds two inputs.
 *
 * This node extends [AddNode] and implements [Differentiable], providing
 * automatic differentiation for addition operations.
 *
 * @param T The type of the values to add.
 * @property add A function that defines how to add two values of type [T].
 */
class DifferentiableAddNode<T>(add: (T, T) -> T) : AddNode<T>(add), Differentiable<T> {
    /**
     * Computes the gradient of this node with respect to the specified input.
     *
     * For addition, the gradient is always 1, because d(a + b)/da = 1 and d(a + b)/db = 1.
     * However, since we don't know how to create a "1" of type T, we just return the
     * gradient of the output with respect to this node.
     *
     * @param inputIndex The index of the input to compute the gradient with respect to.
     * @param gradient The gradient of the output with respect to this node.
     * @return The gradient of this node with respect to the specified input.
     */
    override fun gradient(inputIndex: Int, gradient: T): T {
        // For addition, the gradient is always 1, so we just return the gradient
        return gradient
    }
}