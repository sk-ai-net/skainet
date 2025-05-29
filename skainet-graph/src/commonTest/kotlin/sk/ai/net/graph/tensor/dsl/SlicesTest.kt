package sk.ai.net.graph.tensor.dsl

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.core.TypedTensor
import sk.ai.net.graph.tensor.impl.DoublesTensor
import sk.ai.net.graph.tensor.impl.createTensor
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals

class SlicesTest {

    @Test
    fun `test slice Rank-1 slice`() {

        val tensor = DoublesTensor(
            Shape(3, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        )

        val slicedTensor = slice<Double>(tensor) {
            // from second to the last
            segment {
                all()
            }
            // all elements, equals to ":"
            // from 0 to the second last element
            segment {
                from(1)
            }
        }

        // Since we don't have assertTensorsSimilar yet, we'll just check a few values
        val expected = createTensor(Shape(3, 2), listOf(2.0, 3.0, 5.0, 6.0, 8.0, 9.0).toDoubleArray())

        // Check shape
        assertEquals(2, slicedTensor.shape.dimensions.size)
        assertEquals(3, slicedTensor.shape.dimensions[0])
        assertEquals(2, slicedTensor.shape.dimensions[1])

        // Check a few values
        assertTrue(compareBy(slicedTensor[0, 0], 2.0))
        assertTrue(compareBy(slicedTensor[0, 1], 3.0))
        assertTrue(compareBy(slicedTensor[1, 0], 5.0))
        assertTrue(compareBy(slicedTensor[1, 1], 6.0))
        assertTrue(compareBy(slicedTensor[2, 0], 8.0))
        assertTrue(compareBy(slicedTensor[2, 1], 9.0))
    }

    private fun compareBy(actual: Double, expected: Double): Boolean =
        abs(actual - expected) < 0.0001
}
