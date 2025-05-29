package sk.ai.net.graph.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ExpressionTest {

    @Test
    fun testValueExpression() {
        // Create a value expression
        val expr = value(42)
        
        // Verify that the expression wraps a ValueNode
        assertTrue(expr.node is ValueNode)
        
        // Verify that evaluating the node returns the correct value
        assertEquals(42, expr.node.evaluate())
    }
    
    @Test
    fun testAddExpression() {
        // Create value expressions
        val a = value(5)
        val b = value(7)
        
        // Add the expressions
        val sum = a.plus(b) { x, y -> x + y }
        
        // Verify that the expression wraps an AddNode
        assertTrue(sum.node is AddNode)
        
        // Verify that the AddNode has the correct inputs
        val addNode = sum.node as AddNode<Int>
        assertEquals(2, addNode.inputs.size)
        assertEquals(5, addNode.inputs[0].evaluate())
        assertEquals(7, addNode.inputs[1].evaluate())
        
        // Verify that evaluating the node returns the correct sum
        assertEquals(12, sum.node.evaluate())
    }
    
    @Test
    fun testMultiplyExpression() {
        // Create value expressions
        val a = value(5)
        val b = value(7)
        
        // Multiply the expressions
        val product = a.times(b) { x, y -> x * y }
        
        // Verify that the expression wraps a MultiplyNode
        assertTrue(product.node is MultiplyNode)
        
        // Verify that the MultiplyNode has the correct inputs
        val multiplyNode = product.node as MultiplyNode<Int>
        assertEquals(2, multiplyNode.inputs.size)
        assertEquals(5, multiplyNode.inputs[0].evaluate())
        assertEquals(7, multiplyNode.inputs[1].evaluate())
        
        // Verify that evaluating the node returns the correct product
        assertEquals(35, product.node.evaluate())
    }
    
    @Test
    fun testComplexExpression() {
        // Create value expressions
        val a = value(5)
        val b = value(3)
        val c = value(2)
        val d = value(4)
        
        // Build a complex expression: (a + b) * (c + d)
        val sum1 = a.plus(b) { x, y -> x + y }
        val sum2 = c.plus(d) { x, y -> x + y }
        val result = sum1.times(sum2) { x, y -> x * y }
        
        // Verify the structure of the expression
        assertTrue(result.node is MultiplyNode)
        val multiplyNode = result.node as MultiplyNode<Int>
        
        assertTrue(multiplyNode.inputs[0] is AddNode)
        val addNode1 = multiplyNode.inputs[0] as AddNode<Int>
        
        assertTrue(multiplyNode.inputs[1] is AddNode)
        val addNode2 = multiplyNode.inputs[1] as AddNode<Int>
        
        // Verify the inputs to the add nodes
        assertTrue(addNode1.inputs[0] is ValueNode)
        assertTrue(addNode1.inputs[1] is ValueNode)
        assertEquals(5, addNode1.inputs[0].evaluate())
        assertEquals(3, addNode1.inputs[1].evaluate())
        
        assertTrue(addNode2.inputs[0] is ValueNode)
        assertTrue(addNode2.inputs[1] is ValueNode)
        assertEquals(2, addNode2.inputs[0].evaluate())
        assertEquals(4, addNode2.inputs[1].evaluate())
        
        // Verify the computation: (5 + 3) * (2 + 4) = 8 * 6 = 48
        assertEquals(8, addNode1.evaluate())
        assertEquals(6, addNode2.evaluate())
        assertEquals(48, result.node.evaluate())
    }
    
    @Test
    fun testStringExpressions() {
        // Create string value expressions
        val hello = value("Hello, ")
        val world = value("World!")
        
        // Concatenate the strings
        val greeting = hello.plus(world) { a, b -> a + b }
        
        // Verify that evaluating the node returns the correct concatenation
        assertEquals("Hello, World!", greeting.node.evaluate())
    }
    
    @Test
    fun testDoubleExpressions() {
        // Create double value expressions
        val a = value(2.5)
        val b = value(3.0)
        
        // Add the expressions
        val sum = a.plus(b) { x, y -> x + y }
        
        // Multiply the expressions
        val product = a.times(b) { x, y -> x * y }
        
        // Verify the results
        assertEquals(5.5, sum.node.evaluate(), 0.0001)
        assertEquals(7.5, product.node.evaluate(), 0.0001)
    }
}