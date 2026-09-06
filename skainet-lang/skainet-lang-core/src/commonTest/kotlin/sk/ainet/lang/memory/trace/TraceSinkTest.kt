package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.PlanTensor
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** SKEEP-003 §4.9 / M1-F7 (API half): one event stream, disabled by default, recorded into a ring buffer. */
class TraceSinkTest {

    @Test
    fun noopSinkIsDisabledAndDropsEverything() {
        assertFalse(NoopTraceSink.isEnabled)
        NoopTraceSink.emit(TraceEvent.Counter("rss", 1))
        // helpers short-circuit: the block runs, nothing else happens
        var ran = false
        val r = NoopTraceSink.phase("prefill", 1) { ran = true; 42 }
        assertEquals(42, r); assertTrue(ran)
        assertEquals("k", NoopTraceSink.kernel("matmul", "scalar") { "k" })
    }

    @Test
    fun recordingSinkKeepsOrderAndRingCapacity() {
        val sink = RecordingTraceSink(capacity = 3)
        assertTrue(sink.isEnabled)
        for (i in 1..5) sink.emit(TraceEvent.Counter("c", i.toLong(), timeNanos = i.toLong()))
        val values = sink.eventsOf<TraceEvent.Counter>().map { it.value }
        assertEquals(listOf(3L, 4L, 5L), values)
        assertEquals(5L, sink.emitted); assertEquals(2L, sink.dropped)
        sink.clear()
        assertTrue(sink.events().isEmpty()); assertEquals(0L, sink.emitted); assertEquals(0L, sink.dropped)
        assertFailsWith<IllegalArgumentException> { RecordingTraceSink(0) }
    }

    @Test
    fun phaseHelperEmitsBeginAndEndWithDurationAlsoOnException() {
        val sink = RecordingTraceSink()
        val r = sink.phase("decode", step = 17, attributes = mapOf("tokens" to "1")) { "ok" }
        assertEquals("ok", r)
        val ev = sink.events()
        assertEquals(2, ev.size)
        val begin = assertIs<TraceEvent.PhaseBegin>(ev[0]); val end = assertIs<TraceEvent.PhaseEnd>(ev[1])
        assertEquals("decode", begin.phase); assertEquals(17, begin.step); assertEquals("1", begin.attributes["tokens"])
        assertEquals("decode", end.phase); assertEquals(17, end.step)
        assertTrue(end.durationNanos >= 0); assertTrue(end.timeNanos >= begin.timeNanos)

        sink.clear()
        assertFailsWith<IllegalStateException> { sink.phase("load") { throw IllegalStateException("boom") } }
        assertEquals(listOf("PhaseBegin", "PhaseEnd"), sink.events().map { it::class.simpleName })
    }

    @Test
    fun kernelHelperRecordsOpKernelTensorsAndBytes() {
        val sink = RecordingTraceSink()
        val w = TensorId.parse("model.layers[3].attn.q_proj.weight")
        val out = sink.kernel("matmul", "scalar-q4k", inputs = listOf(null, w), output = TensorId.parse("model.layers[3].attn.q#step=1"), bytesRead = 4096, bytesWritten = 64) { 7 }
        assertEquals(7, out)
        val k = assertIs<TraceEvent.KernelRun>(sink.events().single())
        assertEquals("matmul", k.op); assertEquals("scalar-q4k", k.kernel); assertEquals(listOf(null, w), k.inputs)
        assertEquals(4096L, k.bytesRead); assertEquals(64L, k.bytesWritten); assertTrue(k.durationNanos >= 0)
    }

    @Test
    fun allocationAdapterAndScopeEventsCarryIdentityFields() {
        val sink = RecordingTraceSink()
        val id = TensorId.parse("model.layers[0].mlp.down_proj.weight")
        sink.emit(TraceEvent.Allocation(storageId = 412, scope = ScopeKind.MODEL, bytes = 96L shl 20, origin = id, site = "GgufLoader.kt:120"))
        sink.emit(TraceEvent.AdapterInserted("dequantize", Format(FP32, TensorEncoding.Q6_K), Format.dense(FP32), 96L shl 20, target = id))
        sink.emit(TraceEvent.ScopeReset(ScopeKind.FORWARD, liveBytesBefore = 8L shl 20, liveBytesAfter = 0))
        sink.emit(TraceEvent.Free(412, ScopeKind.MODEL, 96L shl 20))
        val ev = sink.events()
        val a = assertIs<TraceEvent.Allocation>(ev[0]); assertEquals(412L, a.storageId); assertEquals(ScopeKind.MODEL, a.scope); assertEquals(id, a.origin); assertEquals("GgufLoader.kt:120", a.site)
        val ad = assertIs<TraceEvent.AdapterInserted>(ev[1]); assertEquals("dequantize", ad.kind); assertEquals(TensorEncoding.Q6_K, ad.from.encoding); assertTrue(ad.to.isDense); assertEquals(ScopeKind.FORWARD, ad.scope)
        val rs = assertIs<TraceEvent.ScopeReset>(ev[2]); assertEquals(8L shl 20, rs.liveBytesBefore); assertEquals(0L, rs.liveBytesAfter)
        assertIs<TraceEvent.Free>(ev[3])
    }

    @Test
    fun compositeFansOutToEnabledSinksOnly() {
        val a = RecordingTraceSink(); val b = RecordingTraceSink()
        val c = CompositeTraceSink(a, NoopTraceSink, b)
        assertTrue(c.isEnabled)
        c.emit(TraceEvent.Counter("x", 1))
        assertEquals(1, a.events().size); assertEquals(1, b.events().size)
        assertFalse(CompositeTraceSink(NoopTraceSink).isEnabled)
    }

    @Test
    fun memoryPlanEmitsAPlanEvent() {
        val f = Format(FP32, TensorEncoding.Q4_K)
        val input = PlanInput("m", "llama", listOf(PlanTensor("w", TensorId.parse("model.w"), f, 256, 144)), null, 512)
        val plan = MemoryPlans.plan(input, Budget.of(1L shl 30))
        val sink = RecordingTraceSink()
        plan.emit(sink)
        val p = assertIs<TraceEvent.Plan>(sink.events().single())
        assertEquals("m", p.model); assertEquals(512, p.ctx); assertEquals(144L, p.weightsBytes); assertEquals(plan.totalBytes, p.totalBytes)
        assertEquals(1L shl 30, p.budgetBytes); assertEquals(true, p.fits)
        plan.emit(NoopTraceSink) // no-op
        // an Int8 dense format has a distinct string in events
        assertEquals("Int8/Dense(1B)", Format.dense(Int8).toString())
    }

    @Test
    fun clockIsMonotonic() {
        val t0 = TraceClock.nowNanos(); val t1 = TraceClock.nowNanos()
        assertTrue(t1 >= t0)
    }
}
