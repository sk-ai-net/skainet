package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor


class Input(private val inputShape: Shape, override val name: String = "Input") : Module() {

    override val modules: List<Module>
        get() = emptyList()


    override fun forward(input: Tensor): Tensor {
        return input
    }
}