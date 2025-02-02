package sk.ai.net.io.mapper

import sk.ai.net.Tensor
import sk.ai.net.nn.Module

interface ModelValuesMapper {
    fun mapToModel(model: Module, wandb: Map<String, Tensor>)
}