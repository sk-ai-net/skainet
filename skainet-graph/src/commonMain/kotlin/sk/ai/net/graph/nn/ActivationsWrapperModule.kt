package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode

/**
 * A module that applies an activation function to its input.
 *
 * @param T The type of data processed by this module.
 * @param activation The activation function to apply.
 * @param name The name of this module.
 */
class ActivationsWrapperModule<T>(
    private val activation: (T) -> T,
    private val name: String = "activation"
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply the activation function to the input
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }

            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                return activation(inputValue)
            }
        }
    }
}