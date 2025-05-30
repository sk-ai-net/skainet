package sk.ai.net.graph.tensor.dsl

import sk.ai.net.core.tensor.SimpleTensor
import sk.ai.net.core.tensor.shape.Shape
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals

class SlicesTest {

    @Test
    fun `test tensor operations`() {
        // Create a simple tensor
        val tensor = SimpleTensor(
            Shape(3, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        )

        // Test basic tensor operations
        assertEquals(3, tensor.shape.dimensions.size)
        assertEquals(3, tensor.shape.dimensions[0])
        assertEquals(3, tensor.shape.dimensions[1])

        // Check a few values
        assertTrue(compareBy(tensor[0, 0], 1.0))
        assertTrue(compareBy(tensor[0, 1], 2.0))
        assertTrue(compareBy(tensor[0, 2], 3.0))
        assertTrue(compareBy(tensor[1, 0], 4.0))
        assertTrue(compareBy(tensor[1, 1], 5.0))
        assertTrue(compareBy(tensor[1, 2], 6.0))
        assertTrue(compareBy(tensor[2, 0], 7.0))
        assertTrue(compareBy(tensor[2, 1], 8.0))
        assertTrue(compareBy(tensor[2, 2], 9.0))
    }

    private fun compareBy(actual: Double, expected: Double): Boolean =
        abs(actual - expected) < 0.0001
}
