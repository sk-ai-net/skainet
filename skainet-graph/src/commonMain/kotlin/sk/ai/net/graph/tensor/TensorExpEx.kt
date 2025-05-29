package sk.ai.net.graph.tensor

import sk.ai.net.graph.core.AddNode
import sk.ai.net.graph.core.Expr
import sk.ai.net.graph.core.MultiplyNode
import sk.ai.net.graph.core.ValueNode

val Tensor.expr: Expr<Tensor> get() = Expr(ValueNode<Tensor>(this))

operator fun Expr<Tensor>.plus(other: Expr<Tensor>) =
    Expr(AddNode(Tensor::plus).apply {
        inputs += this@plus.node
        inputs += other.node
    })

operator fun Expr<Tensor>.times(other: Expr<Tensor>): Expr<Tensor> =
    Expr(MultiplyNode(Tensor::times).apply {
        inputs += this@times.node
        inputs += other.node
    })

