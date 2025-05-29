package sk.ai.net

import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals

class BroadcastingTest {
    @Test
    fun testBroadcastingAddition() {
        // Test broadcasting a scalar to a vector
        val scalar = DoublesTensor(Shape(), doubleArrayOf(2.0))
        val vector = DoublesTensor(Shape(3), doubleArrayOf(1.0, 2.0, 3.0))
        val result = scalar + vector

        assertEquals(Shape(3), result.shape)
        assertContentEquals(doubleArrayOf(3.0, 4.0, 5.0), (result as DoublesTensor).elements)

        // Test broadcasting in the reverse direction
        val result2 = vector + scalar
        assertEquals(Shape(3), result2.shape)
        assertContentEquals(doubleArrayOf(3.0, 4.0, 5.0), (result2 as DoublesTensor).elements)
    }

    @Test
    fun testBroadcastingSubtraction() {
        // Test broadcasting a scalar to a vector
        val scalar = DoublesTensor(Shape(), doubleArrayOf(5.0))
        val vector = DoublesTensor(Shape(3), doubleArrayOf(1.0, 2.0, 3.0))

        // Scalar - Vector
        val result = scalar - vector
        assertEquals(Shape(3), result.shape)
        assertContentEquals(doubleArrayOf(4.0, 3.0, 2.0), (result as DoublesTensor).elements)

        // Vector - Scalar
        val result2 = vector - scalar
        assertEquals(Shape(3), result2.shape)
        assertContentEquals(doubleArrayOf(-4.0, -3.0, -2.0), (result2 as DoublesTensor).elements)
    }

    @Test
    fun testBroadcastingMatrixOperations() {
        // Create a 2x3 matrix
        val matrix = DoublesTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

        // Create a 1x3 vector (row vector)
        val rowVector = DoublesTensor(Shape(1, 3), doubleArrayOf(10.0, 20.0, 30.0))

        // Test broadcasting the row vector to match the matrix
        val result = matrix + rowVector
        assertEquals(Shape(2, 3), result.shape)
        assertContentEquals(
            doubleArrayOf(11.0, 22.0, 33.0, 14.0, 25.0, 36.0),
            (result as DoublesTensor).elements
        )

        // Create a 2x1 vector (column vector)
        val colVector = DoublesTensor(Shape(2, 1), doubleArrayOf(100.0, 200.0))

        // Test broadcasting the column vector to match the matrix
        val result2 = matrix + colVector
        assertEquals(Shape(2, 3), result2.shape)
        assertContentEquals(
            doubleArrayOf(101.0, 102.0, 103.0, 204.0, 205.0, 206.0),
            (result2 as DoublesTensor).elements
        )
    }

    @Test
    fun testComplexBroadcasting() {
        // Create a 2x1x3 tensor
        val tensor1 = DoublesTensor(
            Shape(2, 1, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )

        // Create a 1x2x1 tensor
        val tensor2 = DoublesTensor(
            Shape(1, 2, 1),
            doubleArrayOf(10.0, 20.0)
        )

        // The result should be a 2x2x3 tensor
        val result = tensor1 + tensor2
        assertEquals(Shape(2, 2, 3), result.shape)
        assertContentEquals(
            doubleArrayOf(
                11.0, 12.0, 13.0, 21.0, 22.0, 23.0,
                14.0, 15.0, 16.0, 24.0, 25.0, 26.0
            ),
            (result as DoublesTensor).elements
        )
    }

    @Test
    fun testBroadcastingPow() {
        // Test broadcasting a scalar to a vector for pow operation
        val scalar = DoublesTensor(Shape(), doubleArrayOf(2.0))
        val vector = DoublesTensor(Shape(3), doubleArrayOf(1.0, 2.0, 3.0))

        // Vector ^ Scalar (each element of vector raised to power of scalar)
        val result = vector.pow(scalar)
        assertEquals(Shape(3), result.shape)
        assertContentEquals(doubleArrayOf(1.0, 4.0, 9.0), (result as DoublesTensor).elements)

        // Test broadcasting with matrices
        val matrix = DoublesTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val rowVector = DoublesTensor(Shape(1, 3), doubleArrayOf(2.0, 1.0, 3.0))

        // Matrix ^ RowVector (broadcasting row vector across matrix rows)
        val result2 = matrix.pow(rowVector)
        assertEquals(Shape(2, 3), result2.shape)
        assertContentEquals(
            doubleArrayOf(1.0, 2.0, 27.0, 16.0, 5.0, 216.0),
            (result2 as DoublesTensor).elements
        )
    }
}
