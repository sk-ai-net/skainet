package sk.ai.net.graph.autodiff

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.AddNode
import sk.ai.net.graph.core.MultiplyNode
import sk.ai.net.graph.core.ValueNode

/**
 * A registry of gradient functions for different types of computation nodes.
 *
 * This class maps ComputeNode types to gradient functions, allowing us to compute
 * gradients for existing node types without extending them.
 */
object GradientRegistry {
    /**
     * A map from ComputeNode class names to gradient functions.
     */
    private val gradientFunctions = mutableMapOf<String, (ComputeNode<*>, Int, Any) -> Any>()

    /**
     * Registers a gradient function for a specific ComputeNode type.
     *
     * @param nodeClass The class of the ComputeNode.
     * @param gradientFunction The gradient function for the node.
     */
    fun <T> registerGradientFunction(
        nodeClass: String,
        gradientFunction: (ComputeNode<T>, Int, T) -> T
    ) {
        @Suppress("UNCHECKED_CAST")
        gradientFunctions[nodeClass] = gradientFunction as (ComputeNode<*>, Int, Any) -> Any
    }

    /**
     * Computes the gradient of a node with respect to the specified input.
     *
     * @param node The node to compute the gradient for.
     * @param inputIndex The index of the input to compute the gradient with respect to.
     * @param gradient The gradient of the output with respect to this node.
     * @return The gradient of the node with respect to the specified input.
     */
    @Suppress("UNCHECKED_CAST")
    fun <T> computeGradient(node: ComputeNode<T>, inputIndex: Int, gradient: T): T {
        val nodeClass = node::class.simpleName ?: throw IllegalArgumentException("Node class has no name")
        val gradientFunction = gradientFunctions[nodeClass] ?: throw IllegalArgumentException("No gradient function registered for node class $nodeClass")
        val result = gradientFunction(node, inputIndex, gradient as Any)
        return result as T
    }

    /**
     * Initializes the registry with gradient functions for the built-in node types.
     */
    init {
        // Register gradient function for AddNode
        registerGradientFunction<Any>(
            AddNode::class.simpleName!!,
            { _, _, gradient -> gradient }
        )

        // Register gradient function for MultiplyNode
        registerGradientFunction<Any>(
            MultiplyNode::class.simpleName!!,
            { node, inputIndex, gradient ->
                val multiplyNode = node as MultiplyNode<Any>
                val otherInput = multiplyNode.inputs[1 - inputIndex].evaluate()
                // For multiplication, the gradient is the other input
                // This assumes that the multiplication operation is commutative
                // and that there's a way to multiply the gradient by the other input
                // In a real implementation, we would need to handle this more carefully
                gradient
            }
        )

        // Register gradient function for ValueNode
        registerGradientFunction<Any>(
            ValueNode::class.simpleName!!,
            { _, _, _ -> throw IllegalArgumentException("Cannot compute gradient for ValueNode") }
        )
    }
}
