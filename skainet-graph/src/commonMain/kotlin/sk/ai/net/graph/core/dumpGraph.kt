package sk.ai.net.graph.core

fun <T> printComputeGraph(node: ComputeNode<T>, indent: String = ""): String =
    when (node) {
        is ValueNode -> "${indent}Value(${node.evaluate()})\n"
        is AddNode -> buildString {
            append("${indent}Add\n")
            node.inputs.forEach { append(printComputeGraph(it, "$indent  ")) }
        }
        is MultiplyNode -> buildString {
            append("${indent}Multiply\n")
            node.inputs.forEach { append(printComputeGraph(it, "$indent  ")) }
        }
        is ActivationNode -> buildString {
            append("${indent}${node}\n")
            node.inputs.forEach { append(printComputeGraph(it, "$indent  ")) }
        }
        else -> "${indent}Unknown\n"
    }
