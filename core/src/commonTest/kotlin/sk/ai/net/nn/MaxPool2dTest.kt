package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.dsl.network
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MaxPool2dTest {
    @Test
    fun max_pool2d_basic() {
        val input = DoublesTensor(
            Shape(1, 1, 4, 4),
            doubleArrayOf(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            )
        )
        val pool = MaxPool2d(kernelSize = 2, stride = 2)
        val result = pool.forward(input) as DoublesTensor
        assertEquals(Shape(1, 1, 2, 2), result.shape)
        assertContentEquals(doubleArrayOf(6.0, 8.0, 14.0, 16.0), result.elements)
    }

    @Test
    fun dsl_support() {
        val module = network {
            input(1)
            maxPool2d {
                kernelSize = 2
                stride = 2
            }
        }
        val mlp = module as sk.ai.net.nn.topology.MLP
        assertTrue(mlp.modules[1] is MaxPool2d)
        val mp = mlp.modules[1] as MaxPool2d
        assertEquals(2, mp.kernelSize)
        assertEquals(2, mp.stride)
    }
}
