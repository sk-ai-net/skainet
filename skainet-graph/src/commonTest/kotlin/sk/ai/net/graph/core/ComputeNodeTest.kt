package sk.ai.net.graph.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.math.exp

class ComputeNodeTest {

    @Test
    fun testValueNode() {
        // Create a ValueNode with an integer value
        val valueNode = ValueNode(42)

        // Verify that evaluate() returns the correct value
        assertEquals(42, valueNode.evaluate())

        // Create a ValueNode with a string value
        val stringNode = ValueNode("test")

        // Verify that evaluate() returns the correct value
        assertEquals("test", stringNode.evaluate())
    }

    @Test
    fun testAddNode() {
        // Create ValueNodes for the inputs
        val value1 = ValueNode(5)
        val value2 = ValueNode(7)

        // Create an AddNode with an integer addition function
        val addNode = AddNode<Int> { a, b -> a + b }
        addNode.inputs.add(value1)
        addNode.inputs.add(value2)

        // Verify that evaluate() returns the correct sum
        assertEquals(12, addNode.evaluate())

        // Test with string concatenation
        val str1 = ValueNode("Hello, ")
        val str2 = ValueNode("World!")

        val strAddNode = AddNode<String> { a, b -> a + b }
        strAddNode.inputs.add(str1)
        strAddNode.inputs.add(str2)

        // Verify string concatenation works
        assertEquals("Hello, World!", strAddNode.evaluate())
    }

    @Test
    fun testMultiplyNode() {
        // Create ValueNodes for the inputs
        val value1 = ValueNode(5)
        val value2 = ValueNode(7)

        // Create a MultiplyNode with an integer multiplication function
        val multiplyNode = MultiplyNode<Int> { a, b -> a * b }
        multiplyNode.inputs.add(value1)
        multiplyNode.inputs.add(value2)

        // Verify that evaluate() returns the correct product
        assertEquals(35, multiplyNode.evaluate())

        // Test with double values
        val double1 = ValueNode(2.5)
        val double2 = ValueNode(3.0)

        val doubleMultiplyNode = MultiplyNode<Double> { a, b -> a * b }
        doubleMultiplyNode.inputs.add(double1)
        doubleMultiplyNode.inputs.add(double2)

        // Verify double multiplication works
        assertEquals(7.5, doubleMultiplyNode.evaluate())
    }

    @Test
    fun testActivationNode() {
        // Create a ValueNode for the input
        val value = ValueNode(0.5)

        // Create an ActivationNode with a ReLU-like function
        val reluNode = ActivationNode<Double>({ if (it > 0) it else 0.0 }, "ReLU")
        reluNode.inputs.add(value)

        // Verify that evaluate() applies the activation function correctly
        assertEquals(0.5, reluNode.inputs[0].evaluate())
        assertEquals(0.5, reluNode.evaluate())

        // Test with a negative value
        val negValue = ValueNode(-0.5)
        val reluNode2 = ActivationNode<Double>({ if (it > 0) it else 0.0 }, "ReLU")
        reluNode2.inputs.add(negValue)

        // Verify ReLU converts negative to zero
        assertEquals(-0.5, reluNode2.inputs[0].evaluate())
        assertEquals(0.0, reluNode2.evaluate())

        // Test with a sigmoid-like function
        val sigmoidNode = ActivationNode<Double>({ 1.0 / (1.0 + exp(-it)) }, "Sigmoid")
        sigmoidNode.inputs.add(value)

        // Verify sigmoid calculation (approximate)
        val expected = 1.0 / (1.0 + exp(-0.5))
        assertEquals(expected, sigmoidNode.evaluate(), 0.0001)
    }

    @Test
    fun testComplexGraph() {
        // Create a more complex computation graph: (5 + 3) * (2 + 4)
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

        // Verify the computation: (5 + 3) * (2 + 4) = 8 * 6 = 48
        assertEquals(8, add1.evaluate())
        assertEquals(6, add2.evaluate())
        assertEquals(48, multiply.evaluate())
    }
}
