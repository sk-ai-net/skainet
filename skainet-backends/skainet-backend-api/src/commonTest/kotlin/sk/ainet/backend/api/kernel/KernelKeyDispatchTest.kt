package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.PackedBlockDecoder
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * SKEEP-003 §5.1 / PRD M1-F5, M1-A4: dispatch keys on declared formats and layouts, rank is
 * normalised once before lookup, adapters are visible, and the reference kernel is correct for any
 * format — so the #993 (rank-1 decode step × packed weight) and #991 (activation subtype) cases
 * cannot crash.
 */
class KernelKeyDispatchTest {

    @AfterTest fun cleanup() { KernelDispatch.clearForTesting() }

    private fun denseView(shape: Shape, init: (Int) -> Float): TensorView {
        val s = Storage.Heap.floats(shape.volume)
        val f = s.floats!!
        for (i in f.indices) f[i] = init(i)
        return TensorView.dense(s, shape, FP32)
    }

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val sign = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return sign; if (e >= 31) return sign or 0x7C00
        return sign or (e shl 10) or (m ushr 13)
    }

    /** A Q8_0 weight of [rows] × 32 with known values: code i - 16, scale 0.5. */
    private fun q8Weight(rows: Int): Pair<TensorView, FloatArray> {
        val bytes = ByteArray(rows * 34)
        for (r in 0 until rows) {
            val off = r * 34; val d = half(0.5f)
            bytes[off] = (d and 0xFF).toByte(); bytes[off + 1] = ((d ushr 8) and 0xFF).toByte()
            for (i in 0 until 32) bytes[off + 2 + i] = ((i - 16) + r).toByte()
        }
        val data = Q8_0BlockTensorData(Shape(rows, 32), bytes)
        val view = TensorView.packed(Storage.Heap.wrap(bytes, mutable = false), Shape(rows, 32), TensorEncoding.Q8_0, PackedBlockDecoder(data), id = TensorId.parse("model.layers[0].attn.q_proj.weight"))
        return view to data.toFloatArray()
    }

    @Test
    fun keysDescribeFormatsAndLayouts() {
        val a = denseView(Shape(1, 32)) { it.toFloat() }
        val (w, _) = q8Weight(2)
        val key = KernelKey.matmul(a, w)
        assertEquals("matmul", key.op); assertEquals(2, key.operands.size)
        assertEquals(OperandKey(Format.dense(FP32), LayoutClass.CONTIGUOUS), key.operands[0])
        assertEquals(
            OperandKey(Format(FP32, TensorEncoding.Q8_0), LayoutClass.BLOCKED_ROW_MAJOR),
            key.operands[1],
            "a weight loaded from a file is canonical, and the key says so (#973)",
        )
        assertEquals("matmul(Float32/Dense(4B) contiguous × Float32/Q8_0 blocked_row_major) @host", key.toString())
        // a strided operand is a different key — that is the point of keying on layout
        val strided = denseView(Shape(4, 8)) { it.toFloat() }.narrow(1, 0, 4)
        assertEquals(LayoutClass.STRIDED, OperandKey.of(strided).layout)
        assertEquals(Format(FP32, TensorEncoding.Q8_0).kernelEncodingName, "Q8_0")
        assertEquals(Format.dense(FP32).kernelEncodingName, "Float32")
    }

    @Test
    fun rankIsNormalisedOnceBeforeLookup() {
        // #993: a rank-1 decode-step activation must never reach a kernel written for rank 2
        val rank1 = denseView(Shape(32)) { 1f }
        val (norm, leading) = KernelDispatch.normalizeActivation(rank1)
        assertEquals(Shape(1, 32), norm.shape); assertTrue(leading.isEmpty())
        assertEquals(rank1.storage, norm.storage)                    // a view, not a copy

        val rank3 = denseView(Shape(2, 3, 8)) { it.toFloat() }
        val (flat, dims) = KernelDispatch.normalizeActivation(rank3)
        assertEquals(Shape(6, 8), flat.shape); assertEquals(listOf(2, 3), dims.toList())
        assertEquals(rank3.storage, flat.storage)
        assertEquals(rank3.get(1, 2, 3), flat.get(5, 3))

        val rank2 = denseView(Shape(4, 8)) { it.toFloat() }
        assertEquals(rank2, KernelDispatch.normalizeActivation(rank2).first)
    }

    @Test
    fun theReferenceKernelIsCorrectForARank1ActivationTimesAPackedWeight() {
        // the exact #993 repro, through the registry: no special-casing, finite output
        val (w, wf) = q8Weight(3)
        val x = denseView(Shape(32)) { (it % 5).toFloat() }
        val (a, _) = KernelDispatch.normalizeActivation(x)
        val out = denseView(Shape(1, 3)) { 0f }
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, sink = sink)

        for (j in 0 until 3) {
            var expect = 0f
            for (t in 0 until 32) expect += (t % 5).toFloat() * wf[j * 32 + t]
            assertTrue(abs(out.get(0, j) - expect) < 1e-3f, "row $j: ${out.get(0, j)} vs $expect")
            assertTrue(out.get(0, j).isFinite())
        }
        val run = assertIs<TraceEvent.KernelRun>(sink.events().single())
        assertEquals("matmul", run.op); assertEquals("reference", run.kernel)
        assertEquals("model.layers[0].attn.q_proj.weight", run.inputs[1]!!.canonical)
    }

    @Test
    fun aRegisteredKernelWinsOverTheReferencePath() {
        val (w, _) = q8Weight(2)
        val a = denseView(Shape(1, 32)) { 1f }
        val out = denseView(Shape(1, 2)) { 0f }
        val key = KernelKey.matmul(a, w)
        var ran = 0
        KernelDispatch.register(object : ViewKernel {
            override val key: KernelKey = key
            override val name: String = "fake-q8"
            override fun run(inputs: List<TensorView>, out: TensorView) { ran++; out.set(0, 0, value = 7f); out.set(0, 1, value = 8f) }
        })
        assertEquals("fake-q8", KernelDispatch.find(key)?.name)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, sink = sink)
        assertEquals(1, ran); assertEquals(7f, out.get(0, 0)); assertEquals(8f, out.get(0, 1))
        assertEquals("fake-q8", assertIs<TraceEvent.KernelRun>(sink.events().single()).kernel)
        assertNull(KernelDispatch.find(KernelKey("softmax", key.operands)))
    }

    @Test
    fun aStridedActivationGetsAVisibleGatherAdapter() {
        val (w, wf) = q8Weight(1)
        val wide = denseView(Shape(2, 64)) { it.toFloat() }
        // two rows of 32 taken from a 64-wide buffer: real gaps between rows
        val a = wide.narrow(1, 0, 32)
        assertTrue(!a.isContiguous, "the activation must be strided for this test to mean anything")
        val out = denseView(Shape(2, 1)) { 0f }
        val scope = ForwardScope(256)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, scope, sink)
        val adapter = assertIs<TraceEvent.AdapterInserted>(sink.events().first { it is TraceEvent.AdapterInserted })
        assertEquals("gather", adapter.kind); assertEquals(64L * 4, adapter.bytes) // 2 x 32 floats gathered
        assertEquals(ScopeKind.FORWARD, scope.kind)
        assertTrue(out.get(0, 0).isFinite() && out.get(1, 0).isFinite())
        // the gathered copy is what the kernel read: same numbers as a manual dot product
        var expect = 0f
        for (t in 0 until 32) expect += a.get(1, t) * wf[t]
        assertTrue(abs(out.get(1, 0) - expect) < 1e-3f, "${out.get(1, 0)} vs $expect")
        scope.close()
    }

    @Test
    fun q4kWeightsAlsoGoThroughTheReferencePath() {
        // #991's shape: the activation is not the subtype the fast path wanted; the reference decodes anyway
        val bytes = ByteArray(144)
        for (i in bytes.indices) bytes[i] = (i * 7).toByte()
        val d = half(0.02f); bytes[0] = (d and 0xFF).toByte(); bytes[1] = ((d ushr 8) and 0xFF).toByte()
        val dmin = half(0.01f); bytes[2] = (dmin and 0xFF).toByte(); bytes[3] = ((dmin ushr 8) and 0xFF).toByte()
        val data = Q4_KBlockTensorData(Shape(1, 256), bytes)
        val w = TensorView.packed(Storage.Heap.wrap(bytes, mutable = false), Shape(1, 256), TensorEncoding.Q4_K, PackedBlockDecoder(data))
        val a = denseView(Shape(1, 256)) { 0.5f }
        val out = denseView(Shape(1, 1)) { 0f }
        KernelDispatch.matmul(a, w, out)
        var expect = 0f
        for (t in 0 until 256) expect += 0.5f * data.toFloatArray()[t]
        assertTrue(abs(out.get(0, 0) - expect) < 1e-2f, "${out.get(0, 0)} vs $expect")
    }
}
