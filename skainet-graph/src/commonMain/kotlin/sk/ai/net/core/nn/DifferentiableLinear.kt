package sk.ai.net.core.nn

import sk.ai.net.graph.autodiff.DifferentiableAddNode
import sk.ai.net.graph.autodiff.DifferentiableMultiplyNode
import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.ValueNode

/**
 * A differentiable linear layer for neural networks.
 *
 * This layer performs a linear transformation of the input: y = Wx + b,
 * where W is the weight matrix, x is the input, and b is the bias.
 * It uses differentiable nodes to enable backpropagation.
 *
 * @param T The type of data processed by this layer.
 * @property weights The weights of the linear transformation.
 * @property bias The bias of the linear transformation.
 * @property multiply A function that defines how to multiply two values of type [T].
 * @property add A function that defines how to add two values of type [T].
 */
class DifferentiableLinear<T>(
    private val weights: T,
    private val bias: T,
    private val multiply: (T, T) -> T,
    private val add: (T, T) -> T,
    override val name: String
) : Module<T> {
    /**
     * Performs a forward pass through the linear layer.
     *
     * @param input The input node.
     * @return The output node after applying the linear transformation.
     */
    override fun forward(input: ComputeNode<T>): ComputeNode<T> =
        DifferentiableAddNode(add).apply {
            inputs += ValueNode(bias)
            inputs += DifferentiableMultiplyNode(multiply).apply {
                inputs += ValueNode(weights)
                inputs += input
            }
        }
}