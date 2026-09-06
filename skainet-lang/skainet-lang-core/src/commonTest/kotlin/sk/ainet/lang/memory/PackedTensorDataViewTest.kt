package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-003 §4.1 façade for the packed encodings, with the constraint that matters most: the view
 * decodes **bit-identically** to the existing block decoders and borrows the very bytes the loader
 * produced (rule 5 — nothing is copied, nothing is re-ordered).
 */
class PackedTensorDataViewTest {

    /** Deterministic bytes with sane FP16 scales, as in the golden parity fixtures. */
    private class Rng(seed: Long) {
        private var s = seed
        fun next(): Long { var x = s; x = x xor (x shl 13); x = x xor (x ushr 7); x = x xor (x shl 17); s = x; return x }
        fun byte(): Byte = (next() ushr 33).toByte()
        fun unit(): Float = ((next() ushr 40).toInt() and 0xFFFF) / 65536f
    }

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val sign = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return sign; if (e >= 31) return sign or 0x7C00
        return sign or (e shl 10) or (m ushr 13)
    }

    private fun le16(b: ByteArray, off: Int, h: Int) { b[off] = (h and 0xFF).toByte(); b[off + 1] = ((h ushr 8) and 0xFF).toByte() }

    private fun blocks(count: Int, bytesPerBlock: Int, seed: Long, scaleOffsets: List<Int>): ByteArray {
        val rng = Rng(seed)
        val b = ByteArray(count * bytesPerBlock) { rng.byte() }
        for (blk in 0 until count) for (off in scaleOffsets) le16(b, blk * bytesPerBlock + off, half(rng.unit() * 0.05f + 0.005f))
        return b
    }

    private fun check(name: String, data: PackedBlockStorage, encoding: TensorEncoding, packedBytes: ByteArray) {
        val v = data.packedView
        // format: logically FP32, physically the block encoding (rule 3)
        assertEquals(Format(FP32, encoding), v.format, "$name format")
        assertFalse(v.format.isDense)
        // zero-copy: the storage borrows the loader's bytes
        assertSame(packedBytes, (v.storage as Storage.Heap).bytes, "$name borrows its bytes")
        assertFalse(v.storage.isMutable, "$name view is read-only")
        // bit-identical decode: block-wise (PackedBlockStorage) vs element-wise (view.get)
        val expected = data.toFloatArray()
        val actual = v.toFloatArray()
        assertEquals(expected.size, actual.size, "$name element count")
        for (i in expected.indices) {
            assertEquals(expected[i].toRawBits(), actual[i].toRawBits(), "$name element $i: ${expected[i]} vs ${actual[i]}")
        }
        // get() decodes, never a raw byte
        assertEquals(expected[0].toRawBits(), v.get(0, 0).toRawBits(), "$name get(0,0)")
        assertFailsWith<IllegalStateException> { v.set(0, 0, value = 1f) }
    }

    @Test fun q4_0() { val b = blocks(4, 18, 1, listOf(0)); check("Q4_0", Q4_0BlockTensorData(Shape(2, 64), b), TensorEncoding.Q4_0, b) }
    @Test fun q5_0() { val b = blocks(4, 22, 2, listOf(0)); check("Q5_0", Q5_0BlockTensorData(Shape(2, 64), b), TensorEncoding.Q5_0, b) }
    @Test fun q5_1() { val b = blocks(4, 24, 3, listOf(0, 2)); check("Q5_1", Q5_1BlockTensorData(Shape(2, 64), b), TensorEncoding.Q5_1, b) }
    @Test fun q8_0() { val b = blocks(4, 34, 4, listOf(0)); check("Q8_0", Q8_0BlockTensorData(Shape(2, 64), b), TensorEncoding.Q8_0, b) }
    @Test fun q4_K() { val b = blocks(2, 144, 5, listOf(0, 2)); check("Q4_K", Q4_KBlockTensorData(Shape(2, 256), b), TensorEncoding.Q4_K, b) }
    @Test fun q5_K() { val b = blocks(2, 176, 6, listOf(0, 2)); check("Q5_K", Q5_KBlockTensorData(Shape(2, 256), b), TensorEncoding.Q5_K, b) }
    @Test fun q6_K() { val b = blocks(2, 210, 7, listOf(208)); check("Q6_K", Q6_KBlockTensorData(Shape(2, 256), b), TensorEncoding.Q6_K, b) }

    @Test
    fun ternary() {
        val values = ByteArray(4 * 32) { ((it % 3) - 1).toByte() }
        val t = Ternary2BitTensorData.fromTernaryValues(Shape(4, 32), values, scale = 0.8125f)
        check("Ternary", t, TensorEncoding.TernaryPacked, t.packedData)
    }

    @Test
    fun packedViewsSliceWholeBlocksOverTheSameBytes() {
        val b = blocks(4, 144, 11, listOf(0, 2))              // 4 Q4_K blocks = [2, 512]
        val data = Q4_KBlockTensorData(Shape(2, 512), b)
        val v = data.packedView
        assertTrue(v.layout.blocked); assertEquals(Shape(2, 2), v.layout.shape)
        val row1 = v.narrow(0, 1, 1)
        assertSame(v.storage, row1.storage)                   // no copy
        val all = data.toFloatArray()
        val expectedRow = all.copyOfRange(512, 1024)
        val actualRow = row1.toFloatArray()
        for (i in expectedRow.indices) assertEquals(expectedRow[i].toRawBits(), actualRow[i].toRawBits(), "row element $i")
        assertFailsWith<IllegalArgumentException> { v.narrow(1, 0, 100) }   // partial block
    }

    @Test
    fun materializingAPackedViewDecodesIntoAScope() {
        val b = blocks(2, 34, 13, listOf(0))
        val data = Q8_0BlockTensorData(Shape(2, 32), b)
        val scope = ForwardScope(128)
        val dense = data.packedView.materialize(scope = scope)
        assertTrue(dense.format.isDense); assertEquals(ScopeKind.FORWARD, dense.storage.scope)
        val expected = data.toFloatArray()
        for (i in expected.indices) assertEquals(expected[i].toRawBits(), dense.toFloatArray()[i].toRawBits())
        scope.close()
    }
}
