package sk.ai.net.io

import sk.ai.net.graph.tensor.Tensor

interface ParametersLoader {
    suspend fun load(onTensorLoaded: (String, Tensor) -> Unit)
}