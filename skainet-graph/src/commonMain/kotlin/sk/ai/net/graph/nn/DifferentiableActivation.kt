package sk.ai.net.graph.nn

import sk.ai.net.graph.autodiff.DifferentiableActivationNode
import sk.ai.net.graph.core.ComputeNode

/**
 * A differentiable module that applies an activation function to its input.
 *
 * This module creates a [DifferentiableActivationNode] to enable backpropagation
 * through activation functions.
 *
 * @param T The type of data processed by this module.
 * @property activationFunc The activation function to apply.
 * @property gradientFunc The function to compute the gradient of the activation function.
 * @property name The name of the activation function, used for debugging.
 */
class DifferentiableActivation<T>(
    private val activationFunc: (T) -> T,
    private val gradientFunc: (T, T) -> T,
    private val name: String = "DifferentiableActivation"
) : Module<T> {
    /**
     * Applies the activation function to the input.
     *
     * Creates a DifferentiableActivationNode with the specified activation function and
     * connects it to the input node.
     *
     * @param input The input computation node.
     * @return The output computation node after applying the activation function.
     */
    override fun forward(input: ComputeNode<T>): ComputeNode<T> =
        DifferentiableActivationNode(activationFunc, gradientFunc, name).apply {
            inputs += input
        }
}