package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals
import kotlin.test.assertNotEquals

class DropoutTest {
    @Test
    fun `dropout in training mode with p=0_5`() {
        val tensor = DoublesTensor(
            Shape(2, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val dropout = Dropout(p = 0.5)
        dropout.train() // Ensure training mode
        
        val result = dropout.forward(tensor) as DoublesTensor
        
        // Check that shape is preserved
        assertEquals(Shape(2, 3), result.shape)
        
        // Check that some elements are zeroed out (this is probabilistic, but with p=0.5 it's very likely)
        // and that non-zero elements are scaled by 1/(1-p) = 2
        var hasZeros = false
        var hasScaledValues = false
        
        for (i in 0 until tensor.size) {
            if (result.elements[i] == 0.0) {
                hasZeros = true
            } else if (result.elements[i] == tensor.elements[i] * 2.0) {
                hasScaledValues = true
            }
        }
        
        // Assert that we have both zeros and scaled values
        // Note: This is a probabilistic test, so there's a very small chance it could fail
        // even if the implementation is correct
        kotlin.test.assertTrue(hasZeros, "Dropout should zero out some elements")
        kotlin.test.assertTrue(hasScaledValues, "Dropout should scale non-zero elements by 1/(1-p)")
    }
    
    @Test
    fun `dropout in evaluation mode`() {
        val tensor = DoublesTensor(
            Shape(2, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val dropout = Dropout(p = 0.5)
        dropout.eval() // Set to evaluation mode
        
        val result = dropout.forward(tensor) as DoublesTensor
        
        // In evaluation mode, dropout should return the input unchanged
        assertEquals(Shape(2, 3), result.shape)
        assertContentEquals(tensor.elements, result.elements)
    }
    
    @Test
    fun `dropout with p=0`() {
        val tensor = DoublesTensor(
            Shape(2, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val dropout = Dropout(p = 0.0)
        dropout.train() // Ensure training mode
        
        val result = dropout.forward(tensor) as DoublesTensor
        
        // With p=0, dropout should return the input unchanged
        assertEquals(Shape(2, 3), result.shape)
        assertContentEquals(tensor.elements, result.elements)
    }
    
    @Test
    fun `dropout with p=1`() {
        val tensor = DoublesTensor(
            Shape(2, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        val dropout = Dropout(p = 1.0)
        dropout.train() // Ensure training mode
        
        val result = dropout.forward(tensor) as DoublesTensor
        
        // With p=1, dropout should zero out all elements
        assertEquals(Shape(2, 3), result.shape)
        assertContentEquals(DoubleArray(tensor.size) { 0.0 }, result.elements)
    }
    
    @Test
    fun `dropout preserves tensor shape`() {
        val shapes = listOf(
            Shape(1, 10),
            Shape(5, 5),
            Shape(2, 3, 4),
            Shape(1, 2, 3, 4)
        )
        
        for (shape in shapes) {
            val tensor = DoublesTensor(shape, DoubleArray(shape.volume) { 1.0 })
            val dropout = Dropout(p = 0.5)
            dropout.train()
            
            val result = dropout.forward(tensor) as DoublesTensor
            
            assertEquals(shape, result.shape, "Dropout should preserve tensor shape")
        }
    }
}