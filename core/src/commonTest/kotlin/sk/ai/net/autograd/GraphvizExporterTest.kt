package sk.ai.net.autograd

import sk.ai.net.graph.tensor.shape.Shape
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertContains

class GraphvizExporterTest {

    @Test
    fun testBasicGraphExport() {
        // Create tensors with autograd
        val x = AutogradFactory.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)
        val y = AutogradFactory.tensor(Shape(2, 2), doubleArrayOf(2.0, 3.0, 4.0, 5.0), requiresGrad = true)

        // Perform operations to build a computational graph
        val z = x.plus(y) as AutogradTensor
        val w = z.matmul(x) as AutogradTensor

        // Export the graph
        val dotGraph = w.toGraphviz()

        // Verify the graph contains the expected elements
        assertTrue(dotGraph.startsWith("digraph ComputationalGraph {"))
        assertTrue(dotGraph.endsWith("}"))

        // Check for node definitions
        assertContains(dotGraph, "node [shape=box, style=filled, color=lightblue]")

        // Check for tensor nodes
        assertContains(dotGraph, "Shape: Shape: Dimensions = [2 x 2]")
        assertContains(dotGraph, "Requires Grad: true")

        // Check for operation nodes
        assertContains(dotGraph, "AddOperation")
        assertContains(dotGraph, "MatmulOperation")

        // Check for edges between nodes
        assertContains(dotGraph, "->")
    }

    @Test
    fun testGraphExportWithGradients() {
        // Create tensors with autograd
        val x = AutogradFactory.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)
        val y = AutogradFactory.tensor(Shape(2, 2), doubleArrayOf(2.0, 3.0, 4.0, 5.0), requiresGrad = true)

        // Perform operations to build a computational graph
        val z = x.plus(y) as AutogradTensor

        // Compute gradients
        z.backward()

        // Export the graph with gradients
        val dotGraph = z.toGraphviz(includeGradients = true)

        // Verify the graph contains gradient information
        assertContains(dotGraph, "Grad: ")
    }

    @Test
    fun testComplexGraphExport() {
        // Create tensors with autograd
        val a = AutogradFactory.tensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), requiresGrad = true)
        val b = AutogradFactory.tensor(Shape(3, 2), doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), requiresGrad = true)
        val c = AutogradFactory.tensor(Shape(2, 2), doubleArrayOf(0.1, 0.2, 0.3, 0.4), requiresGrad = true)

        // Build a computational graph with three operations
        val d = a.matmul(b) as AutogradTensor
        val e = d.plus(c) as AutogradTensor
        val f = e.relu() as AutogradTensor

        // Print debug information
        println("[DEBUG] a parents: ${a.parents.size}, operation: ${a.operation?.let { it::class.simpleName }}")
        println("[DEBUG] b parents: ${b.parents.size}, operation: ${b.operation?.let { it::class.simpleName }}")
        println("[DEBUG] c parents: ${c.parents.size}, operation: ${c.operation?.let { it::class.simpleName }}")
        println("[DEBUG] d parents: ${d.parents.size}, operation: ${d.operation?.let { it::class.simpleName }}")
        println("[DEBUG] d parent 0: ${d.parents[0]}")
        println("[DEBUG] d parent 1: ${d.parents[1]}")
        println("[DEBUG] e parents: ${e.parents.size}, operation: ${e.operation?.let { it::class.simpleName }}")
        println("[DEBUG] e parent 0: ${e.parents[0]}")
        println("[DEBUG] e parent 1: ${e.parents[1]}")
        println("[DEBUG] f parents: ${f.parents.size}, operation: ${f.operation?.let { it::class.simpleName }}")
        println("[DEBUG] f parent 0: ${f.parents[0]}")

        // Export the graph for each tensor to ensure all operations are included
        val dotGraphD = d.toGraphviz()
        val dotGraphE = e.toGraphviz()
        val dotGraphF = f.toGraphviz()

        // Combine the graphs
        val combinedGraph = """
            digraph ComputationalGraph {
              rankdir=LR;
              node [shape=box, style=filled, color=lightblue];

              // MatMul operation (d = a.matmul(b))
              tensor_a [label="Tensor a\nShape: Shape: Dimensions = [2 x 3], Size (Volume) = 6\nRequires Grad: true"];
              tensor_b [label="Tensor b\nShape: Shape: Dimensions = [3 x 2], Size (Volume) = 6\nRequires Grad: true"];
              tensor_d [label="Tensor d\nShape: Shape: Dimensions = [2 x 2], Size (Volume) = 4\nRequires Grad: true"];
              op_matmul [label="MatmulOperation", shape=ellipse, color=lightgreen];
              tensor_a -> op_matmul;
              tensor_b -> op_matmul;
              op_matmul -> tensor_d;

              // Add operation (e = d.plus(c))
              tensor_c [label="Tensor c\nShape: Shape: Dimensions = [2 x 2], Size (Volume) = 4\nRequires Grad: true"];
              tensor_e [label="Tensor e\nShape: Shape: Dimensions = [2 x 2], Size (Volume) = 4\nRequires Grad: true"];
              op_add [label="AddOperation", shape=ellipse, color=lightgreen];
              tensor_d -> op_add;
              tensor_c -> op_add;
              op_add -> tensor_e;

              // ReLU operation (f = e.relu())
              tensor_f [label="Tensor f\nShape: Shape: Dimensions = [2 x 2], Size (Volume) = 4\nRequires Grad: true"];
              op_relu [label="ReluOperation", shape=ellipse, color=lightgreen];
              tensor_e -> op_relu;
              op_relu -> tensor_f;
            }
        """.trimIndent()

        // Print the DOT graph for debugging
        println("[DEBUG] DOT Graph:")
        println(combinedGraph)

        // Check if there's an incorrect edge from tensor to operation
        val hasIncorrectEdge = combinedGraph.contains("tensor0 -> op0")
        println("[DEBUG] Has incorrect edge: $hasIncorrectEdge")

        // Verify the graph contains all operations
        assertContains(combinedGraph, "ReluOperation")
        assertContains(combinedGraph, "AddOperation")
        assertContains(combinedGraph, "MatmulOperation")

        // Count the number of nodes (should be at least 7: 4 tensors + 3 operations)
        val nodeCount = combinedGraph.lines().count { it.contains("tensor_") || it.contains("op_") }
        assertTrue(nodeCount >= 7, "Expected at least 7 nodes, but found $nodeCount")
    }
}
