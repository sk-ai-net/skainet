package sk.ai.net

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertContentEquals

class ShapeTest {

    @Test
    fun volumeIsProductOfDimensions() {
        val shape = Shape(2, 3, 4)
        assertEquals(24, shape.volume)
    }

    @Test
    fun rankIsNumberOfDimensions() {
        val shape = Shape(2, 3)
        assertEquals(2, shape.rank)
    }

    @Test
    fun equalityChecksAllDimensions() {
        val shape1 = Shape(2, 3)
        val shape2 = Shape(2, 3)
        val shape3 = Shape(3, 2)
        assertEquals(shape1, shape2)
        assertFalse(shape1 == shape3)
    }

    @Test
    fun constructorCopiesDimensions() {
        val dims = intArrayOf(2, 3)
        val shape = Shape(*dims)
        dims[0] = 5
        assertContentEquals(intArrayOf(2, 3), shape.dimensions)
    }

    @Test
    fun toStringContainsSizeInformation() {
        val shape = Shape(2, 3)
        assertEquals("Shape: Dimensions = [2 x 3], Size (Volume) = 6", shape.toString())
    }
}
