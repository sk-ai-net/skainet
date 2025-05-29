package sk.ai.net.nn.reflection

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module

import kotlin.test.Test
import kotlin.test.assertEquals

class ModuleToolsTest {

    val expected = """
        ParentModule
        ├── SubModule1
        ├── SubModule2
        └── SubModule3
            └── SubSubModule1

    """.trimIndent()

    internal class SimpleModule(
        override val name: String,
        override val modules: List<Module> = emptyList()
    ) : Module() {
        override fun forward(input: Tensor): Tensor {
            // A dummy forward implementation
            return input
        }

    }

    @Test
    fun `visual module shows hierarchy`() {
        // Create a tree of modules
        val subModule1 = SimpleModule("SubModule1")
        val subModule2 = SimpleModule("SubModule2")
        val subSubModule1 = SimpleModule("SubSubModule1")
        val subModule3 = SimpleModule("SubModule3", listOf(subSubModule1))

        val parentModule = SimpleModule("ParentModule", listOf(subModule1, subModule2, subModule3))

        // Visualize the hierarchy
        val visualString = parentModule.toVisualString()
        println(visualString)
        assertEquals(visualString, expected)
    }
}