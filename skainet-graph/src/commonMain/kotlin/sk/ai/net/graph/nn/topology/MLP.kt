package sk.ai.net.graph.nn.topology

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.nn.Module

/**
 * A multi-layer perceptron.
 *
 * @param T The type of data processed by this module.
 * @param modules The modules that make up the MLP.
 */
class MLP<T>(
    private vararg val modules: Module<T>
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply each module in sequence
        var current = input
        for (module in modules) {
            current = module.forward(current)
        }
        return current
    }
}