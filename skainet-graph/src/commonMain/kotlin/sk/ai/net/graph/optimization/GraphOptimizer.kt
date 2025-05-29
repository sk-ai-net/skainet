package sk.ai.net.graph.optimization

import sk.ai.net.graph.core.ComputeNode

/**
 * Interface for graph optimization passes.
 *
 * A graph optimizer transforms a computation graph to improve its performance
 * or reduce its memory usage. Examples of optimizations include constant folding,
 * common subexpression elimination, and operator fusion.
 *
 * @param T The type of data processed by the computation nodes.
 */
interface GraphOptimizer<T> {
    /**
     * Optimizes a computation graph.
     *
     * @param node The root node of the computation graph to optimize.
     * @return The optimized computation graph.
     */
    fun optimize(node: ComputeNode<T>): ComputeNode<T>
}