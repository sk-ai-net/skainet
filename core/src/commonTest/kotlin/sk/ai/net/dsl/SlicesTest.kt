package sk.ai.net.dsl

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.core.TypedTensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.impl.createTensor
import sk.ai.net.utils.assertTensorsSimilar
import kotlin.math.abs
import kotlin.test.Test

class SlicesTest {

    @Test
    fun `test slice Rank-1 slice`() {

        val tensor = DoublesTensor(
            Shape(3, 3),
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        )

        val slicedTensor = slice(tensor) {
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
        assertTensorsSimilar(
            createTensor(Shape(3, 2), listOf(2, 3, 5, 6, 8, 9).toIntArray()) as TypedTensor<Double>,
            slicedTensor,
            message = "Sliced tensor is not as expected",
            ::compareBy
        )
    }

    fun compareBy(actual: Double, expected: Double): Boolean =
        abs(actual - expected) < 0.0001
}