package sk.ai.net.core.nn

import sk.ai.net.graph.core.AddNode
import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.MultiplyNode
import sk.ai.net.graph.core.ValueNode

class Linear<T>(
    private val weights: T,
    private val bias: T,
    private val matmul: (T, T) -> T,
    private val add: (T, T) -> T,
    override val name: String = "Linear",
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> =
        AddNode(add).apply {
            inputs += ValueNode(bias)
            inputs += MultiplyNode(matmul).apply {
                inputs += ValueNode(weights)
                inputs += input
            }
        }
}