package sk.ai.net.nn.activations

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module

class ActivationsWrapperModule(private val activationHandler: (Tensor) -> Tensor, override val name: String) :
    Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        return activationHandler(input)
    }
}