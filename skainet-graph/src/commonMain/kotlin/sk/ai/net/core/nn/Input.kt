package sk.ai.net.core.nn

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.ValueNode
import sk.ai.net.core.tensor.shape.Shape

/**
 * An input layer.
 *
 * @param T The type of data processed by this module.
 * @param shape The shape of the input.
 * @param name The name of this module.
 * @param createTensor The function to create a tensor with the given shape.
 */
class Input<T>(
    private val shape: Shape,
    override val name: String = "input",
    private val createTensor: (Shape) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Return the input as is
        return input
    }

    /**
     * Creates a compute node with a tensor of the given shape.
     *
     * @return A compute node with a tensor of the given shape.
     */
    fun createNode(): ComputeNode<T> {
        val tensor = createTensor(shape)
        return ValueNode(tensor)
    }
}