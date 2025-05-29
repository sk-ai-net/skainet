package sk.ai.net.nn.activations

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module


class ReLU(override val name: String = "ReLU") : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor = input.relu()
}

