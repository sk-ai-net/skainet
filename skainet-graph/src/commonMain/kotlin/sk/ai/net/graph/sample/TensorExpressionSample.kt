package sk.ai.net.graph.sample

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.expr
import sk.ai.net.graph.tensor.plus
import sk.ai.net.graph.tensor.times

fun createTensorExpressionSample(aValue: Double, bValue: Double, cValue: Double): ComputeNode<Tensor> {
    // Create tensors with single value
    val a = SimpleTensor(listOf(1), doubleArrayOf(aValue))
    val b = SimpleTensor(listOf(1), doubleArrayOf(bValue))
    val c = SimpleTensor(listOf(1), doubleArrayOf(cValue))

    val d = (a.expr + b.expr) * c.expr
    return d.node

}
