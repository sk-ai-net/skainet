package sk.ai.net.core.nn

import sk.ai.net.graph.core.ActivationNode
import sk.ai.net.graph.core.ComputeNode

/**
 * A module that applies an activation function to its input.
 *
 * Activation functions are commonly used in neural networks to introduce
 * non-linearity into the model. Common activation functions include ReLU,
 * Sigmoid, and Tanh.
 *
 * @param T The type of data processed by this module.
 * @property activationFunc The activation function to apply.
 * @property name The name of the activation function, used for debugging.
 */
class Activation<T>(
    private val activationFunc: (T) -> T,
    override val name: String = "Activation"
) : Module<T> {
    /**
     * Applies the activation function to the input.
     *
     * Creates an ActivationNode with the specified activation function and
     * connects it to the input node.
     *
     * @param input The input computation node.
     * @return The output computation node after applying the activation function.
     */
    override fun forward(input: ComputeNode<T>): ComputeNode<T> =
        ActivationNode(activationFunc, name).apply {
            inputs += input
        }
}
