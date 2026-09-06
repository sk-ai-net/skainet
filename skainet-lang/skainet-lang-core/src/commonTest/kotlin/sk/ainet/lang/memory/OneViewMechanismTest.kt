@file:Suppress("DEPRECATION") // half of this test's job is to compare against the deprecated mechanisms

package sk.ainet.lang.memory

import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.slice
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * #1034 (SKEEP-003 §4.6): **one** view mechanism.
 *
 * Slicing used to be an index remapper (`SlicedTensorView`), aliasing a byte range
 * (`BufferHandle.Aliased`) and a packed transpose a byte permutation — three unrelated ways of
 * looking at someone else's bytes. This asserts that `TensorView` + `Layout` produces the same
 * answers as each of them, over the same `Storage`, and that every view operation returns the same
 * type so they compose.
 */
class OneViewMechanismTest {

    private fun rampTensor(vararg dims: Int): Tensor<FP32, Float> {
        lateinit var built: Tensor<FP32, Float>
        data {
            built = tensor<FP32, Float> {
                shape(*dims) { shape ->
                    init { indices ->
                        var flat = 0
                        var stride = 1
                        for (i in indices.indices.reversed()) { flat += indices[i] * stride; stride *= shape[i] }
                        flat.toFloat()
                    }
                }
            }
        }
        return built
    }

    private fun valuesOf(t: Tensor<FP32, Float>): List<Float> {
        val out = ArrayList<Float>(t.shape.volume)
        val idx = IntArray(t.shape.rank)
        repeat(t.shape.volume) { flat ->
            var rem = flat
            for (d in t.shape.rank - 1 downTo 0) { idx[d] = rem % t.shape[d]; rem /= t.shape[d] }
            out += t.data.get(*idx)
        }
        return out
    }

    // --- one mechanism, one type -------------------------------------------------------------

    @Test
    fun everyViewOperationReturnsAViewOverTheSameStorage() {
        val t = rampTensor(4, 6)
        val v = t.view()
        val derived = listOf(
            "narrow" to v.narrow(1, 2, 4),
            "transpose" to v.transpose(),
            "unsqueeze" to v.unsqueeze(0),
            "step" to v.step(1, 2),
            "slice" to v.slice(Slice.Range<FP32, Float>(1, 3), Slice.All<FP32, Float>()),
            "squeeze" to v.narrow(0, 1, 1).squeeze(0),
        )
        for ((name, d) in derived) {
            assertEquals(v.storage.id, d.storage.id, "$name must be zero-copy over the same Storage")
            assertEquals(v.format, d.format, "$name must not change what the values mean")
        }
    }

    @Test
    fun sliceComposesFromNarrowStepAndSqueeze() {
        val t = rampTensor(4, 6)
        val v = t.view()
        val bySlice = v.slice(Slice.At<FP32, Float>(2), Slice.Step<FP32, Float>(1, 6, 2))
        val byHand = v.narrow(0, 2, 1).narrow(1, 1, 5).step(1, 2).squeeze(0)
        assertEquals(byHand.shape, bySlice.shape)
        assertTrue(byHand.toFloatArray().contentEquals(bySlice.toFloatArray()))
    }

    // --- subsumes SlicedTensorView (index remap) ----------------------------------------------

    @Test
    fun theViewAgreesWithTheOldSlicedTensorViewOnEverySliceKind() {
        val t = rampTensor(4, 3, 2)
        val cases: List<Pair<String, List<Slice<FP32, Float>>>> = listOf(
            "all" to listOf(Slice.All(), Slice.All(), Slice.All()),
            "range on axis 0" to listOf(Slice.Range(1, 3), Slice.All(), Slice.All()),
            "range on two axes" to listOf(Slice.Range(1, 4), Slice.Range(0, 2), Slice.All()),
            "at on axis 0" to listOf(Slice.At(2), Slice.All(), Slice.All()),
            "at on axis 1" to listOf(Slice.All(), Slice.At(1), Slice.All()),
            "two ats" to listOf(Slice.At(3), Slice.At(0), Slice.All()),
            "step" to listOf(Slice.Step(0, 4, 2), Slice.All(), Slice.All()),
            "step and range" to listOf(Slice.Step(0, 4, 3), Slice.Range(1, 3), Slice.At(1)),
        )
        for ((name, slices) in cases) {
            val old = t.slice(slices)
            val new = t.view().slice(slices)
            assertEquals(old.shape, new.shape, "$name: shape")
            assertTrue(
                valuesOf(old).toFloatArray().contentEquals(new.toFloatArray()),
                "$name: values — old ${valuesOf(old)} vs new ${new.toFloatArray().toList()}",
            )
        }
    }

    @Test
    fun aSliceOfASliceIsStillOneView() {
        val t = rampTensor(6, 8)
        val root = t.view()
        val once = root.narrow(0, 1, 4)
        val twice = once.narrow(1, 2, 4).step(0, 2)
        assertEquals(Shape(2, 4), twice.shape)
        assertEquals(root.storage.id, twice.storage.id, "composition never copies")
        // the same elements the old mechanism would have reached
        val expected = t.slice(listOf(Slice.Step<FP32, Float>(1, 5, 2), Slice.Range<FP32, Float>(2, 6)))
        assertTrue(valuesOf(expected).toFloatArray().contentEquals(twice.toFloatArray()))
    }

    // --- subsumes BufferHandle.Aliased (byte range) -------------------------------------------

    @Test
    fun aByteRangeAliasIsAViewOverASlicedStorage() {
        val floats = FloatArray(64) { it.toFloat() }
        val parent = Storage.Heap.wrap(floats)
        val region = parent.slice(offsetBytes = 16 * 4, lengthBytes = 16 * 4)
        val view = TensorView.dense(region, Shape(4, 4))

        assertTrue(region.owner is Owner.Alias, "an alias declares its parent, like BufferHandle.Aliased did")
        assertEquals(16f, view.get(0, 0), "the region starts where the alias starts")
        assertEquals(31f, view.get(3, 3))
        assertEquals(64L * 4, parent.sizeBytes)
        assertEquals(16L * 4, region.sizeBytes)

        // shared memory, exactly as Aliased promised: a write through the parent is visible here
        floats[16] = -1f
        assertEquals(-1f, view.get(0, 0))

        // and the same bounds contract
        assertFailsWith<IllegalArgumentException> { parent.slice(60L * 4, 8L * 4) }

        // the deprecated handle described the same region, and nothing more
        val handle = BufferHandle.Aliased(BufferHandle.Borrowed(ByteArray(64 * 4)), 16L * 4, 16L * 4)
        assertEquals(handle.sizeInBytes, region.sizeBytes, "the same region, described the old way")
        assertFailsWith<IllegalArgumentException> {
            BufferHandle.Aliased(BufferHandle.Borrowed(ByteArray(64 * 4)), 60L * 4, 8L * 4)
        }
    }

    // --- the layout is the mechanism ----------------------------------------------------------

    @Test
    fun eachViewCallWrapsTheSameBytesInAFreshHandle() {
        // `view()` is a *handle* over the tensor's array, so two calls carry two StorageIds — but
        // one array: a write through either is visible through the other. Callers that compare
        // identity (tracing, plan-vs-actual) must hold on to one view rather than re-deriving it.
        val t = rampTensor(2, 3)
        val a = t.view()
        val b = t.view()
        assertNotEquals(a.storage.id, b.storage.id, "a new handle per call")
        a.set(0, 0, value = -7f)
        assertEquals(-7f, b.get(0, 0), "the same bytes underneath")
        assertEquals(-7f, t.data.get(0, 0), "and the same bytes the TensorData reads")
    }

    @Test
    fun transposeIsMetadataOnly() {
        val t = rampTensor(3, 5)
        val v = t.view()
        val tv = v.transpose()
        assertEquals(Shape(5, 3), tv.shape)
        assertEquals(v.storage.id, tv.storage.id)
        assertNotEquals(v.layout.strides.toList(), tv.layout.strides.toList())
        for (r in 0 until 3) for (c in 0 until 5) assertEquals(v.get(r, c), tv.get(c, r), "($r,$c)")
        assertTrue(!tv.isContiguous, "a transposed view is strided, not a copy")
    }

    @Test
    fun stepIsAStrideMultiply() {
        val t = rampTensor(8)
        val v = t.view().step(0, 3)
        assertEquals(Shape(3), v.shape)
        assertTrue(floatArrayOf(0f, 3f, 6f).contentEquals(v.toFloatArray()))
        assertEquals(3, v.layout.strides[0])
        assertFailsWith<IllegalArgumentException> { v.step(0, 0) }
    }
}
