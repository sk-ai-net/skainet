package sk.ai.net.graph.tensor

import sk.ai.net.graph.core.ComputeNode


class TensorValueNode(val tensor: Tensor) : ComputeNode<Tensor>() {
    override fun evaluate() = tensor
}

class TensorAddNode : ComputeNode<Tensor>() {
    override fun evaluate(): Tensor {
        val left = inputs[0].evaluate()
        val right = inputs[1].evaluate()
        return left + right
    }
}

class TensorMultiplyNode : ComputeNode<Tensor>() {
    override fun evaluate(): Tensor {
        val left = inputs[0].evaluate()
        val right = inputs[1].evaluate()
        return left * right
    }
}
