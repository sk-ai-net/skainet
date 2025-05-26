package sk.ai.net

import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.math.abs
import kotlin.math.exp

class SoftmaxTest {
    @Test
    fun testSoftmaxBasic() {
        // Create a simple vector
        val vector = DoublesTensor(Shape(3), doubleArrayOf(1.0, 2.0, 3.0))
        
        // Apply softmax
        val result = vector.softmax()
        
        // Calculate expected values manually
        val expValues = doubleArrayOf(exp(1.0), exp(2.0), exp(3.0))
        val sum = expValues.sum()
        val expected = expValues.map { it / sum }.toDoubleArray()
        
        // Check that the result is close to the expected values
        assertEquals(Shape(3), result.shape)
        for (i in expected.indices) {
            assertTrue(abs((result as DoublesTensor).elements[i] - expected[i]) < 1e-10)
        }
        
        // Check that the sum is approximately 1
        val resultSum = (result as DoublesTensor).elements.sum()
        assertTrue(abs(resultSum - 1.0) < 1e-10)
    }
    
    @Test
    fun testSoftmaxNumericalStability() {
        // Create a vector with large values that could cause overflow
        val vector = DoublesTensor(Shape(3), doubleArrayOf(1000.0, 1000.0, 1000.0))
        
        // Apply softmax
        val result = vector.softmax()
        
        // All values should be approximately equal (close to 1/3)
        assertEquals(Shape(3), result.shape)
        for (i in 0 until 3) {
            assertTrue(abs((result as DoublesTensor).elements[i] - 1.0/3.0) < 1e-10)
        }
        
        // Check that the sum is approximately 1
        val resultSum = (result as DoublesTensor).elements.sum()
        assertTrue(abs(resultSum - 1.0) < 1e-10)
    }
    
    @Test
    fun testSoftmaxWithDimension() {
        // Create a 2x3 matrix
        val matrix = DoublesTensor(
            Shape(2, 3), 
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
        
        // Apply softmax along dimension 1 (columns)
        val result = matrix.softmax(1)
        
        // Calculate expected values manually for each row
        val expValues1 = doubleArrayOf(exp(1.0), exp(2.0), exp(3.0))
        val sum1 = expValues1.sum()
        val expected1 = expValues1.map { it / sum1 }.toDoubleArray()
        
        val expValues2 = doubleArrayOf(exp(4.0), exp(5.0), exp(6.0))
        val sum2 = expValues2.sum()
        val expected2 = expValues2.map { it / sum2 }.toDoubleArray()
        
        // Check that the result is close to the expected values
        assertEquals(Shape(2, 3), result.shape)
        
        // Check first row
        for (i in 0 until 3) {
            assertTrue(abs((result as DoublesTensor).elements[i] - expected1[i]) < 1e-10)
        }
        
        // Check second row
        for (i in 0 until 3) {
            assertTrue(abs((result as DoublesTensor).elements[i + 3] - expected2[i]) < 1e-10)
        }
        
        // Check that the sum of each row is approximately 1
        val row1Sum = (result as DoublesTensor).elements.slice(0 until 3).sum()
        val row2Sum = (result as DoublesTensor).elements.slice(3 until 6).sum()
        assertTrue(abs(row1Sum - 1.0) < 1e-10)
        assertTrue(abs(row2Sum - 1.0) < 1e-10)
    }
    
    @Test
    fun testSoftmaxWithDimensionNumericalStability() {
        // Create a 2x3 matrix with large values
        val matrix = DoublesTensor(
            Shape(2, 3), 
            doubleArrayOf(1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0)
        )
        
        // Apply softmax along dimension 1 (columns)
        val result = matrix.softmax(1)
        
        // Check that the result has the expected shape
        assertEquals(Shape(2, 3), result.shape)
        
        // For the first row, all values should be approximately equal (close to 1/3)
        for (i in 0 until 3) {
            assertTrue(abs((result as DoublesTensor).elements[i] - 1.0/3.0) < 1e-10)
        }
        
        // For the second row, all values should be approximately equal (close to 1/3)
        for (i in 3 until 6) {
            assertTrue(abs((result as DoublesTensor).elements[i] - 1.0/3.0) < 1e-10)
        }
        
        // Check that the sum of each row is approximately 1
        val row1Sum = (result as DoublesTensor).elements.slice(0 until 3).sum()
        val row2Sum = (result as DoublesTensor).elements.slice(3 until 6).sum()
        assertTrue(abs(row1Sum - 1.0) < 1e-10)
        assertTrue(abs(row2Sum - 1.0) < 1e-10)
    }
}