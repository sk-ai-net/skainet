package sk.ai.net.graph.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LazyComputeNodeTest {

    // A test compute node that counts the number of times it's evaluated
    class CountingNode(private val value: Int) : ComputeNode<Int>() {
        var evaluationCount = 0
            private set
        
        override fun evaluate(): Int {
            evaluationCount++
            return value
        }
    }
    
    @Test
    fun testLazyEvaluation() {
        // Create a counting node
        val countingNode = CountingNode(42)
        
        // Create a lazy node wrapping the counting node
        val lazyNode = LazyComputeNode(countingNode)
        
        // Verify that the counting node hasn't been evaluated yet
        assertEquals(0, countingNode.evaluationCount)
        
        // Evaluate the lazy node
        val result = lazyNode.evaluate()
        
        // Verify that the result is correct
        assertEquals(42, result)
        
        // Verify that the counting node was evaluated once
        assertEquals(1, countingNode.evaluationCount)
        
        // Evaluate the lazy node again
        val result2 = lazyNode.evaluate()
        
        // Verify that the result is still correct
        assertEquals(42, result2)
        
        // Verify that the counting node wasn't evaluated again
        assertEquals(1, countingNode.evaluationCount)
    }
    
    @Test
    fun testClearCache() {
        // Create a counting node
        val countingNode = CountingNode(42)
        
        // Create a lazy node wrapping the counting node
        val lazyNode = LazyComputeNode(countingNode)
        
        // Evaluate the lazy node
        lazyNode.evaluate()
        
        // Verify that the counting node was evaluated once
        assertEquals(1, countingNode.evaluationCount)
        
        // Clear the cache
        lazyNode.clearCache()
        
        // Evaluate the lazy node again
        lazyNode.evaluate()
        
        // Verify that the counting node was evaluated again
        assertEquals(2, countingNode.evaluationCount)
    }
    
    @Test
    fun testExtensionFunction() {
        // Create a counting node
        val countingNode = CountingNode(42)
        
        // Create a lazy node using the extension function
        val lazyNode = countingNode.lazy()
        
        // Verify that the lazy node is a LazyComputeNode
        assertTrue(lazyNode is LazyComputeNode)
        
        // Evaluate the lazy node
        val result = lazyNode.evaluate()
        
        // Verify that the result is correct
        assertEquals(42, result)
        
        // Verify that the counting node was evaluated once
        assertEquals(1, countingNode.evaluationCount)
        
        // Evaluate the lazy node again
        val result2 = lazyNode.evaluate()
        
        // Verify that the result is still correct
        assertEquals(42, result2)
        
        // Verify that the counting node wasn't evaluated again
        assertEquals(1, countingNode.evaluationCount)
    }
    
    @Test
    fun testComplexGraph() {
        // Create counting nodes
        val a = CountingNode(5)
        val b = CountingNode(3)
        
        // Create an add node
        val addNode = AddNode<Int> { x, y -> x + y }
        addNode.inputs.add(a)
        addNode.inputs.add(b)
        
        // Create a lazy node wrapping the add node
        val lazyAddNode = addNode.lazy()
        
        // Verify that the counting nodes haven't been evaluated yet
        assertEquals(0, a.evaluationCount)
        assertEquals(0, b.evaluationCount)
        
        // Evaluate the lazy node
        val result = lazyAddNode.evaluate()
        
        // Verify that the result is correct
        assertEquals(8, result)
        
        // Verify that the counting nodes were each evaluated once
        assertEquals(1, a.evaluationCount)
        assertEquals(1, b.evaluationCount)
        
        // Evaluate the lazy node again
        val result2 = lazyAddNode.evaluate()
        
        // Verify that the result is still correct
        assertEquals(8, result2)
        
        // Verify that the counting nodes weren't evaluated again
        assertEquals(1, a.evaluationCount)
        assertEquals(1, b.evaluationCount)
        
        // Clear the cache
        lazyAddNode.clearCache()
        
        // Evaluate the lazy node again
        val result3 = lazyAddNode.evaluate()
        
        // Verify that the result is still correct
        assertEquals(8, result3)
        
        // Verify that the counting nodes were evaluated again
        assertEquals(2, a.evaluationCount)
        assertEquals(2, b.evaluationCount)
    }
}