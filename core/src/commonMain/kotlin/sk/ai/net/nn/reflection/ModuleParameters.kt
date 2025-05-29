package sk.ai.net.nn.reflection

import sk.ai.net.graph.tensor.Tensor

sealed class ModuleParameter {
    abstract val name: String
    abstract var value: Tensor

    data class WeightParameter(
        override val name: String,
        override var value: Tensor
    ) : ModuleParameter()

    data class BiasParameter(
        override val name: String,
        override var value: Tensor
    ) : ModuleParameter()
}

interface ModuleParameters {
    val params: List<ModuleParameter>
}

public fun List<ModuleParameter>.by(name: String): ModuleParameter? =
    firstOrNull { namedParameter -> namedParameter.name.uppercase().contains(name.uppercase()) }

// Returns the first BiasParameter or throws a NoSuchElementException if none is found.
fun List<ModuleParameter>.bias(): ModuleParameter.BiasParameter =
    this.filterIsInstance<ModuleParameter.BiasParameter>()
        .firstOrNull() ?: throw NoSuchElementException("No bias parameter found!")

// Returns the first WeightParameter or throws a NoSuchElementException if none is found.
fun List<ModuleParameter>.weights(): ModuleParameter.WeightParameter =
    this.filterIsInstance<ModuleParameter.WeightParameter>()
        .firstOrNull() ?: throw NoSuchElementException("No weight parameter found!")
