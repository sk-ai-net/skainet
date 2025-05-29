package sk.ai.net.nn.topology

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module
import sk.ai.net.nn.reflection.ModuleParameter
import sk.ai.net.nn.reflection.ModuleParameters

class MLP(vararg modules: Module, override val name: String = "FeedForwardNetwork") : Module(), ModuleParameters {
    private val modulesList = modules.toList()
    override val modules: List<Module>
        get() = modulesList

    override fun forward(input: Tensor): Tensor {
        var tmp = input
        modulesList.forEach { module ->
            tmp = module.forward(tmp)
        }
        return tmp
    }

    override val params: List<ModuleParameter>
        get() = emptyList()
}