package sk.ai.net.nn.activations

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module

class Softmax(private val dimension: Int, override val name: String = "Softmax") : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        return input.softmax(dimension)
    }
}

