package sk.ai.net.graph.optimization

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.ValueNode

/**
 * A graph optimizer that performs constant folding.
 *
 * Constant folding is an optimization technique that evaluates constant expressions
 * at compile time rather than at runtime. For example, the expression `2 + 3` can
 * be replaced with the constant `5`.
 *
 * @param T The type of data processed by the computation nodes.
 */
class ConstantFoldingOptimizer<T> : GraphOptimizer<T> {
    /**
     * Optimizes a computation graph by folding constant expressions.
     *
     * @param node The root node of the computation graph to optimize.
     * @return The optimized computation graph.
     */
    override fun optimize(node: ComputeNode<T>): ComputeNode<T> {
        // If the node has no inputs, it's already a leaf node (e.g., a ValueNode)
        if (node.inputs.isEmpty()) {
            return node
        }

        // Recursively optimize the inputs
        val optimizedInputs = node.inputs.map { optimize(it) }

        // If all inputs are ValueNodes, we can evaluate the node at compile time
        if (optimizedInputs.all { it is ValueNode<*> }) {
            // Evaluate the node with the optimized inputs
            val result = node.evaluate()
            return ValueNode(result)
        }

        // Otherwise, create a new node with the optimized inputs
        node.inputs.clear()
        node.inputs.addAll(optimizedInputs)
        return node
    }
}