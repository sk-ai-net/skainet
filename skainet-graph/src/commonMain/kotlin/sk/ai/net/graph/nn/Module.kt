package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode

/**
 * Interface representing a neural network module.
 *
 * A module is a building block of a neural network that transforms an input
 * computation node into an output computation node. Modules can be composed
 * to create complex neural network architectures.
 *
 * @param T The type of data processed by this module.
 */
interface Module<T> {
    /**
     * Applies the module to the input computation node.
     *
     * This method defines the forward pass of the module, transforming the input
     * computation node into an output computation node.
     *
     * @param input The input computation node.
     * @return The output computation node.
     */
    fun forward(input: ComputeNode<T>): ComputeNode<T>
}
