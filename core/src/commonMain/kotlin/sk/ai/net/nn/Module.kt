package sk.ai.net.nn

import sk.ai.net.graph.tensor.Tensor


abstract class Module {

    abstract val name: String

    abstract val modules: List<Module>

    abstract fun forward(input: Tensor): Tensor

    operator fun invoke(input: Tensor): Tensor {
        return forward(input)
    }

    //abstract fun createTensor(descriptor: DataDescriptor, shape: Shape, list: List<Any?>): Tensor<T>
}

