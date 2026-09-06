package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-003 rules 4–6: `get()` decodes (never a raw byte), views are zero-copy over the same
 * storage, `materialize()` is the only copy point.
 */
class TensorViewTest {

    private fun denseView(vararg dims: Int): TensorView {
        val shape = Shape(*dims)
        val s = Storage.Heap.floats(shape.volume)
        val f = s.floats!!
        for (i in f.indices) f[i] = i.toFloat()
        return TensorView.dense(s, shape, FP32, TensorId.parse("model.w"))
    }

    @Test
    fun denseViewReadsAndWritesElements() {
        val v = denseView(2, 3)
        assertEquals(Format.dense(FP32), v.format); assertTrue(v.isContiguous); assertTrue(v.isMutable)
        assertEquals(0f, v.get(0, 0)); assertEquals(4f, v.get(1, 1)); assertEquals(5f, v.get(1, 2))
        v.set(0, 1, value = 9f)
        assertEquals(9f, v.get(0, 1))
        assertContentEquals(floatArrayOf(0f, 9f, 2f, 3f, 4f, 5f), v.toFloatArray())
        assertFailsWith<IllegalArgumentException> { v.get(2, 0) }
        assertTrue(v.toString().contains("model.w")); assertTrue(v.toString().contains("Float32/Dense"))
    }

    @Test
    fun viewsShareStorageAndAreZeroCopy() {
        val v = denseView(4, 4)
        val rows = v.narrow(0, 1, 2)
        assertEquals(Shape(2, 4), rows.shape); assertSame(v.storage, rows.storage)
        assertEquals(4f, rows.get(0, 0)); assertEquals(11f, rows.get(1, 3))
        rows.set(0, 0, value = -1f)
        assertEquals(-1f, v.get(1, 0))                      // same bytes
        val cols = v.narrow(1, 2, 2)
        assertEquals(Shape(4, 2), cols.shape); assertFalse(cols.isContiguous)
        assertEquals(2f, cols.get(0, 0)); assertEquals(7f, cols.get(1, 1))
        val t = v.transpose()
        assertEquals(Shape(4, 4), t.shape); assertEquals(v.get(1, 2), t.get(2, 1))
        val u = v.narrow(0, 0, 1).squeeze(0)
        assertEquals(Shape(4), u.shape); assertEquals(-1f, v.get(1, 0))
        assertEquals(Shape(1, 4), u.unsqueeze(0).shape)
        assertEquals("model.w[1..3)]", rows.id!!.canonical) // TensorId.view brackets the range
    }

    @Test
    fun packedViewDecodesElementsNeverRawBytes() {
        // a Q8_0 weight: 2 rows x 32 elements = 2 blocks of 34 bytes
        val bytes = ByteArray(2 * 34)
        fun half(v: Float): Int { val b = v.toRawBits(); val s = (b ushr 16) and 0x8000; val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF; return if (e <= 0) s else if (e >= 31) s or 0x7C00 else s or (e shl 10) or (m ushr 13) }
        for (blk in 0 until 2) {
            val off = blk * 34
            val d = half(0.5f); bytes[off] = (d and 0xFF).toByte(); bytes[off + 1] = ((d ushr 8) and 0xFF).toByte()
            for (i in 0 until 32) bytes[off + 2 + i] = (i - 16).toByte()
        }
        val packed = Q8_0BlockTensorData(Shape(2, 32), bytes)
        val storage = Storage.Heap.wrap(bytes)
        val v = TensorView.packed(storage, Shape(2, 32), TensorEncoding.Q8_0, PackedBlockDecoder(packed), id = TensorId.parse("model.layers[0].attn.q_proj.weight"))

        assertEquals(Format(FP32, TensorEncoding.Q8_0), v.format)
        assertFalse(v.format.isDense)
        // rule 4: a decoded float, not the raw code byte
        assertEquals(-16 * 0.5f, v.get(0, 0)); assertEquals(15 * 0.5f, v.get(0, 31)); assertEquals(-16 * 0.5f, v.get(1, 0))
        assertContentEquals(packed.toFloatArray(), v.toFloatArray())
        // writing through a packed view is refused
        assertFailsWith<IllegalStateException> { v.set(0, 0, value = 1f) }
    }

    @Test
    fun packedViewsSliceWholeBlocksWithoutTouchingBytes() {
        val bytes = ByteArray(4 * 144)                       // 4 blocks of Q4_K
        val packed = Q4_KBlockTensorData(Shape(2, 512), bytes)
        val v = TensorView.packed(Storage.Heap.wrap(bytes), Shape(2, 512), TensorEncoding.Q4_K, PackedBlockDecoder(packed))
        assertTrue(v.layout.blocked); assertEquals(Shape(2, 2), v.layout.shape)  // 2 rows x 2 blocks
        val row = v.narrow(0, 1, 1)
        assertEquals(Shape(1, 512), row.shape); assertSame(v.storage, row.storage)
        assertEquals(2L, row.layout.offsetElements)          // two blocks in
        val half = v.narrow(1, 256, 256)                     // one block wide
        assertEquals(Shape(2, 256), half.shape); assertEquals(1L, half.layout.offsetElements)
        assertFailsWith<IllegalArgumentException> { v.narrow(1, 0, 100) }   // not a whole block
        assertFailsWith<IllegalArgumentException> { v.narrow(1, 100, 256) } // block-unaligned start
    }

    @Test
    fun materializeIsTheOnlyCopyPoint() {
        val v = denseView(2, 2)
        val scope = ForwardScope(64)
        val m = v.materialize(scope = scope)
        assertTrue(m.format.isDense); assertEquals(ScopeKind.FORWARD, m.storage.scope)
        assertContentEquals(v.toFloatArray(), m.toFloatArray())
        m.set(0, 0, value = 42f)
        assertEquals(0f, v.get(0, 0))                        // a copy, not a view
        // packed → dense materialization decodes
        val bytes = ByteArray(144)
        val packed = Q4_KBlockTensorData(Shape(1, 256), bytes)
        val pv = TensorView.packed(Storage.Heap.wrap(bytes), Shape(1, 256), TensorEncoding.Q4_K, PackedBlockDecoder(packed))
        val dense = pv.materialize()
        assertTrue(dense.format.isDense)
        assertTrue(abs(dense.get(0, 5) - packed.toFloatArray()[5]) < 1e-6f)
        scope.close()
    }

    @Test
    fun viewsOverClosedStorageAreRefused() {
        val s = Storage.Heap.floats(4)
        val v = TensorView.dense(s, Shape(4))
        s.close()
        assertFailsWith<StorageClosedException> { v.get(0) }
        assertFailsWith<StorageClosedException> { v.toFloatArray() }
        assertFailsWith<IllegalArgumentException> { TensorView.dense(s, Shape(4)) }
    }
}
