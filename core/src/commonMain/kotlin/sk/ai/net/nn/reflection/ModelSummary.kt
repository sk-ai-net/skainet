package sk.ai.net.nn.reflection

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.impl.prod
import sk.ai.net.nn.Module
import sk.ai.net.nn.reflection.table.table

data class NodeSummary(val name: String, val input: Shape, val output: Shape, val params: Long)

class Summary {

    val nodes = mutableListOf<NodeSummary>()

    private fun nodeSummary(index: Int, module: Module, input: Shape, output: Tensor): NodeSummary {
        var params = 0L

        if (module is ModuleParameters) {

            module.params.by("W")?.let { weight ->
                val dimension =
                    DoublesTensor(weight.value.shape, weight.value.shape.dimensions.map { it.toDouble() }.toDoubleArray())
                params += dimension.prod().toLong()
            }

            module.params.by("B")?.let { bias ->
                val dimension =
                    DoublesTensor(bias.value.shape, bias.value.shape.dimensions.map { it.toDouble() }.toDoubleArray())
                params += dimension.prod().toLong()
            }
        }


        return NodeSummary(
            module.name,
            input,
            output.shape,
            params
        )
    }


    fun summary(model: Module, input: Shape, batch_size: Int = -1): List<NodeSummary> {
        var data = DoublesTensor(input, List(input.volume) { 0.0 }.toDoubleArray())
        var count = 1
        model.modules.forEach { module ->
            val moduleInput = data
            data = module.forward(moduleInput) as DoublesTensor
            val nodeSummary = nodeSummary(count, module, moduleInput.shape, data)
            if (nodeSummary.params > 0) {
                count++
                nodes.add(nodeSummary)
            }
        }
        return nodes
    }

    fun printSummary(nodes: List<NodeSummary>) =
        table {
            cellStyle {
                border = true
            }
            header {
                row {
                    cell("Layer (type)")
                    cell("Output Shape")
                    cell("Param #")
                }
            }
            nodes.forEach { node ->
                row {
                    cell(node.name)
                    cell(node.output.toString())
                    cell(node.params)
                }
            }
        }.toString()
}

fun Module.summary(input: Shape, batch_size: Int = -1): String {
    val summary = Summary()
    val nodes = summary.summary(this, input, batch_size)
    return summary.printSummary(nodes)
}
