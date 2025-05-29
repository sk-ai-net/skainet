package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode

/**
 * A module that applies a sequence of modules in order.
 *
 * The Sequential module is a container for a linear stack of modules.
 * Data is passed through all modules in the order they were provided.
 * This is a common pattern in neural networks, where data flows through
 * layers in a sequential manner.
 *
 * @param T The type of data processed by this module.
 * @property modules The modules to apply in sequence.
 */
class Sequential<T>(vararg val modules: Module<T>) : Module<T> {
    /**
     * Applies each module in sequence to the input.
     *
     * The output of each module is passed as input to the next module.
     *
     * @param input The input computation node.
     * @return The output computation node after passing through all modules.
     */
    override fun forward(input: ComputeNode<T>): ComputeNode<T> =
        modules.fold(input) { node, module -> module.forward(node) }
}
