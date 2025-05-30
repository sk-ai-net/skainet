package sk.ai.net.core.double

import sk.ai.net.graph.core.AddNode
import sk.ai.net.graph.core.Expr
import sk.ai.net.graph.core.MultiplyNode
import sk.ai.net.graph.core.ValueNode
import kotlin.collections.plusAssign

val Double.expr: Expr<Double> get() = Expr(ValueNode(this))

operator fun Expr<Double>.plus(other: Expr<Double>) =
    Expr(AddNode(Double::plus).apply {
        inputs += this@plus.node
        inputs += other.node
    })

operator fun Expr<Double>.times(other: Expr<Double>): Expr<Double> =
    Expr(MultiplyNode(Double::times).apply {
        inputs += this@times.node
        inputs += other.node
    })

