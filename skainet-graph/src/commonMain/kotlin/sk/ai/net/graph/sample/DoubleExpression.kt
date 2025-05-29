package sk.ai.net.graph.sample

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.double.expr
import sk.ai.net.graph.double.plus
import sk.ai.net.graph.double.times

fun createStaticGraph(): ComputeNode<Double> {
    val a = 3.0.expr
    val b = 2.0.expr
    val c = (a + b) * 2.0.expr

    val result = c.node.evaluate()
    println("Result: $result")
    return c.node
}