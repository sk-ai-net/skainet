package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** SKEEP-003 §0 *Layout*: strides, offset and contiguity of a view — metadata only. */
class LayoutTest {

    private val f32 = Format.dense(FP32)

    @Test
    fun rowMajorStridesOffsetsAndContiguity() {
        val l = Layout.rowMajor(Shape(2, 3, 4), f32)
        assertContentEquals(intArrayOf(12, 4, 1), l.strides)
        assertTrue(l.isRowMajor); assertTrue(l.isContiguous)
        assertEquals(4, l.elementBytes); assertEquals(24L, l.elementCount); assertEquals(0L, l.offsetBytes)
        assertEquals(17L, l.indexOf(1, 1, 1)); assertEquals(68L, l.byteOffsetOf(1, 1, 1))
        assertFailsWith<IllegalArgumentException> { l.indexOf(2, 0, 0) }
        assertFailsWith<IllegalArgumentException> { l.indexOf(0, 0) }
    }

    @Test
    fun narrowKeepsStridesAndShiftsTheOffset() {
        val l = Layout.rowMajor(Shape(4, 8), f32)
        val n = l.narrow(axis = 0, from = 1, size = 2)
        assertEquals(Shape(2, 8), n.shape); assertContentEquals(intArrayOf(8, 1), n.strides)
        assertEquals(8L, n.offsetElements); assertEquals(32L, n.offsetBytes)
        assertTrue(n.isContiguous)
        val cols = l.narrow(axis = 1, from = 2, size = 3)
        assertEquals(Shape(4, 3), cols.shape); assertEquals(2L, cols.offsetElements)
        assertFalse(cols.isContiguous) // a column slice has gaps
        assertFailsWith<IllegalArgumentException> { l.narrow(1, 6, 4) }
    }

    @Test
    fun transposeSwapsExtentsAndStridesWithoutTouchingBytes() {
        val l = Layout.rowMajor(Shape(2, 3), f32)
        val t = l.transpose()
        assertEquals(Shape(3, 2), t.shape); assertContentEquals(intArrayOf(1, 3), t.strides)
        assertEquals(l.offsetElements, t.offsetElements)
        assertFalse(t.isContiguous); assertFalse(t.isRowMajor)
        assertEquals(l, t.transpose()) // involution
        assertEquals(l.indexOf(1, 2), t.indexOf(2, 1))
    }

    @Test
    fun unsqueezeAndSqueezeAreInverse() {
        val l = Layout.rowMajor(Shape(4), f32)
        val u = l.unsqueeze(0)
        assertEquals(Shape(1, 4), u.shape); assertContentEquals(intArrayOf(0, 1), u.strides); assertTrue(u.isContiguous)
        assertEquals(l, u.squeeze(0))
        assertEquals(Shape(4, 1), l.unsqueeze(1).shape)
        assertFailsWith<IllegalArgumentException> { Layout.rowMajor(Shape(2, 3), f32).squeeze(0) }
    }

    @Test
    fun blockedLayoutAddressesBlocksNotElements() {
        // Q4_K: 256 elements per block, 144 bytes per block; a [4, 512] weight is [4, 2] blocks
        val b = Layout.blocked(Shape(4, 512), blockSize = 256, bytesPerBlock = 144)
        assertTrue(b.blocked); assertEquals(Shape(4, 2), b.shape)
        assertContentEquals(intArrayOf(2, 1), b.strides); assertEquals(144, b.elementBytes)
        assertEquals(144L * 5, b.byteOffsetOf(2, 1))
        val row = b.narrow(0, 1, 1)
        assertEquals(2L, row.offsetElements); assertEquals(144L * 2, row.offsetBytes)
        assertFailsWith<IllegalArgumentException> { Layout.blocked(Shape(4, 300), 256, 144) } // not a whole number of blocks
    }

    @Test
    fun equalityAndRendering() {
        val a = Layout.rowMajor(Shape(2, 2), f32); val b = Layout.rowMajor(Shape(2, 2), f32)
        assertEquals(a, b); assertEquals(a.hashCode(), b.hashCode())
        assertTrue(a.toString().contains("contiguous"))
        assertTrue(a.transpose().toString().contains("strided"))
    }
}
