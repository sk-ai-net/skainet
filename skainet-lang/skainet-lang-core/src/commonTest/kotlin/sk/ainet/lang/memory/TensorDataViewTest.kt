package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.LazyZeroFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-003 §4.1: `TensorData` becomes a façade over `TensorView` — the view is over the *same*
 * bytes (borrowed, zero-copy), reads agree with the data's own accessors, and nothing is copied.
 */
class TensorDataViewTest {

    @Test
    fun denseFloatDataExposesAZeroCopyView() {
        val buf = FloatArray(6) { it.toFloat() }
        val data = DenseFloatArrayTensorData<FP32>(Shape(2, 3), buf)
        val v = assertNotNull(data.view)
        assertEquals(Format.dense(FP32), v.format); assertEquals(Shape(2, 3), v.shape); assertTrue(v.isContiguous)
        // same bytes, no copy: the storage borrows the array
        assertSame(buf, (v.storage as Storage.Heap).floats)
        assertEquals(ScopeKind.AMBIENT, v.storage.scope)
        // reads agree
        for (i in 0 until 2) for (j in 0 until 3) assertEquals(data.get(i, j), v.get(i, j))
        // writes are visible through both
        data.set(1, 2, value = 42f); assertEquals(42f, v.get(1, 2))
        v.set(0, 0, value = -1f); assertEquals(-1f, data.get(0, 0)); assertEquals(-1f, buf[0])
        assertContentEquals(data.copyToFloatArray(), v.toFloatArray())
    }

    @Test
    fun denseIntDataExposesAnInt32View() {
        val buf = IntArray(4) { it * 10 }
        val data = DenseIntArrayTensorData<Int32>(Shape(4), buf)
        val v = assertNotNull(data.view)
        assertEquals(Format.dense(Int32), v.format)
        assertSame(buf, (v.storage as Storage.Heap).ints)
        assertEquals(20f, v.get(2))
    }

    @Test
    fun lazyZeroDataMaterializesThroughTheView() {
        val data = LazyZeroFloatArrayTensorData<FP32>(Shape(2, 2))
        val v = assertNotNull(data.view)
        assertContentEquals(FloatArray(4), v.toFloatArray())
        data.set(1, 1, value = 5f)
        assertEquals(5f, assertNotNull(data.view).get(1, 1)) // the view is over the materialized buffer
    }

    @Test
    fun narrowFloatDataDecodesThroughTheView() {
        // BF16: the high 16 bits of the float
        fun bf16(v: Float): Int = (v.toRawBits() ushr 16) and 0xFFFF
        val values = floatArrayOf(1f, -2.5f, 0.5f, 100f)
        val bytes = ByteArray(values.size * 2)
        for ((i, x) in values.withIndex()) { val b = bf16(x); bytes[i * 2] = (b and 0xFF).toByte(); bytes[i * 2 + 1] = ((b ushr 8) and 0xFF).toByte() }
        val data = Bf16DenseTensorData(Shape(4), bytes)
        val v = assertNotNull(data.view)
        assertEquals(BF16, v.format.dtype); assertEquals(2, v.layout.elementBytes)
        for (i in values.indices) assertEquals(data.get(i), v.get(i), "element $i")
        assertContentEquals(data.copyToFloatArray(), v.toFloatArray())
        assertSame(bytes, (v.storage as Storage.Heap).bytes)

        val fp16 = Fp16DenseTensorData(Shape(2), ByteArray(4) { (it * 17).toByte() })
        val fv = assertNotNull(fp16.view)
        assertEquals(FP16, fv.format.dtype)
        assertEquals(fp16.get(0), fv.get(0)); assertEquals(fp16.get(1), fv.get(1))
    }

    @Test
    fun dataWithoutAViewReportsNull() {
        val anonymous = object : TensorData<FP32, Float> {
            override val shape: Shape = Shape(1)
            override fun get(vararg indices: Int): Float = 0f
            override fun set(vararg indices: Int, value: Float) {}
        }
        assertNull(anonymous.view)
        assertNull(anonymous.encoding)
    }
}
