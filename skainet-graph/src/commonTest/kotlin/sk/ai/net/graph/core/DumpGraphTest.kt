package sk.ai.net.graph.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DumpGraphTest {

    @Test
    fun testPrintValueNode() {
        // Create a value node
        val valueNode = ValueNode(42)
        
        // Print the graph
        val output = printComputeGraph(valueNode)
        
        // Verify the output
        assertEquals("Value(42)\n", output)
    }
    
    @Test
    fun testPrintAddNode() {
        // Create an add node with two value nodes
        val a = ValueNode(5)
        val b = ValueNode(7)
        
        val addNode = AddNode<Int> { x, y -> x + y }
        addNode.inputs.add(a)
        addNode.inputs.add(b)
        
        // Print the graph
        val output = printComputeGraph(addNode)
        
        // Verify the output
        val expected = """
            Add
              Value(5)
              Value(7)
        """.trimIndent() + "\n"
        
        assertEquals(expected, output)
    }
    
    @Test
    fun testPrintMultiplyNode() {
        // Create a multiply node with two value nodes
        val a = ValueNode(5)
        val b = ValueNode(7)
        
        val multiplyNode = MultiplyNode<Int> { x, y -> x * y }
        multiplyNode.inputs.add(a)
        multiplyNode.inputs.add(b)
        
        // Print the graph
        val output = printComputeGraph(multiplyNode)
        
        // Verify the output
        val expected = """
            Multiply
              Value(5)
              Value(7)
        """.trimIndent() + "\n"
        
        assertEquals(expected, output)
    }
    
    @Test
    fun testPrintActivationNode() {
        // Create an activation node with a value node
        val a = ValueNode(0.5)
        
        val activationNode = ActivationNode<Double>({ if (it > 0) it else 0.0 }, "ReLU")
        activationNode.inputs.add(a)
        
        // Print the graph
        val output = printComputeGraph(activationNode)
        
        // Verify the output
        val expected = """
            ReLU
              Value(0.5)
        """.trimIndent() + "\n"
        
        assertEquals(expected, output)
    }
    
    @Test
    fun testPrintComplexGraph() {
        // Create a complex graph: (5 + 3) * (2 + 4)
        val a = ValueNode(5)
        val b = ValueNode(3)
        val c = ValueNode(2)
        val d = ValueNode(4)
        
        // (5 + 3)
        val add1 = AddNode<Int> { x, y -> x + y }
        add1.inputs.add(a)
        add1.inputs.add(b)
        
        // (2 + 4)
        val add2 = AddNode<Int> { x, y -> x + y }
        add2.inputs.add(c)
        add2.inputs.add(d)
        
        // (5 + 3) * (2 + 4)
        val multiply = MultiplyNode<Int> { x, y -> x * y }
        multiply.inputs.add(add1)
        multiply.inputs.add(add2)
        
        // Print the graph
        val output = printComputeGraph(multiply)
        
        // Verify the output
        val expected = """
            Multiply
              Add
                Value(5)
                Value(3)
              Add
                Value(2)
                Value(4)
        """.trimIndent() + "\n"
        
        assertEquals(expected, output)
    }
    
    @Test
    fun testPrintWithCustomIndent() {
        // Create a value node
        val valueNode = ValueNode(42)
        
        // Print the graph with custom indentation
        val output = printComputeGraph(valueNode, "-->")
        
        // Verify the output
        assertEquals("-->Value(42)\n", output)
    }
    
    @Test
    fun testPrintWithLazyNode() {
        // Create a lazy node wrapping a value node
        val valueNode = ValueNode(42)
        val lazyNode = valueNode.lazy()
        
        // Print the graph
        val output = printComputeGraph(lazyNode)
        
        // Verify the output contains "Unknown" since LazyComputeNode is not explicitly handled
        assertTrue(output.contains("Unknown"))
    }
}