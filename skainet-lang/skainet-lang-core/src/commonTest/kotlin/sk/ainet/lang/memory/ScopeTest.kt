package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-003 §4.5: Forward (recycled per step), Model (closed with the model), Ambient (GC) — M1-F3 / M1-F4. */
class ScopeTest {

    @Test
    fun ambientIsTheDefaultAndUntracked() {
        assertEquals(ScopeKind.AMBIENT, Scope.Ambient.kind)
        val s = Scope.Ambient.allocateFloats(8)
        assertEquals(ScopeKind.AMBIENT, s.scope); assertEquals(0L, Scope.Ambient.liveBytes)
        Scope.Ambient.close() // no-op
        assertTrue(s.isAlive)
        assertIs<Storage>(Scope.Ambient.allocate(16, MemoryDomain.HOST_HEAP))
    }

    @Test
    fun forwardScopeBumpAllocatesAndResetRecyclesWithZeroSlabAllocations() {
        val sink = RecordingTraceSink()
        val f = ForwardScope(slabFloats = 1024, sink = sink)
        val slabAllocs = sink.eventsOf<TraceEvent.Allocation>().size // the slab itself
        assertEquals(1, slabAllocs)
        var last: Storage.Heap? = null
        repeat(3) { step ->
            val a = f.allocateFloats(256); val b = f.allocateFloats(512)
            assertEquals(768, f.usedFloats); assertEquals(768L * 4, f.liveBytes)
            assertEquals(ScopeKind.FORWARD, a.scope); assertIs<Owner.Alias>(a.owner) // views over the slab
            a.floats!![a.arrayOffset] = step.toFloat(); b.floats!![b.arrayOffset + 511] = 1f
            assertSame(a.floats, b.floats) // same slab array
            assertEquals(0, a.arrayOffset); assertEquals(256, b.arrayOffset)
            last = a
            f.reset()
            assertEquals(0, f.usedFloats); assertFalse(a.isAlive); assertFalse(b.isAlive)
            assertFailsWith<StorageClosedException> { a.checkAlive() }
        }
        assertEquals(3L, f.steps); assertEquals(768, f.peakFloats)
        // steady state: no new Allocation events after the slab — only the per-step ScopeReset events
        assertEquals(1, sink.eventsOf<TraceEvent.Allocation>().size)
        assertEquals(3, sink.eventsOf<TraceEvent.ScopeReset>().size)
        assertEquals(768L * 4, sink.eventsOf<TraceEvent.ScopeReset>().first().liveBytesBefore)
        f.close(); assertTrue(f.isClosed)
        assertFailsWith<IllegalStateException> { f.allocateFloats(1) }
        assertFalse(last!!.isAlive)
    }

    @Test
    fun forwardOverflowIsCountedAndFreedAtReset() {
        val sink = RecordingTraceSink()
        val f = ForwardScope(slabFloats = 100, sink = sink)
        val a = f.allocateFloats(80)              // in the slab
        val o = f.allocateFloats(50)              // does not fit → overflow storage
        assertIs<Owner.Owned>(o.owner); assertEquals(200L, f.overflowBytes); assertEquals(80, f.usedFloats)
        assertEquals(80L * 4 + 200L, f.liveBytes)
        assertEquals(2, sink.eventsOf<TraceEvent.Allocation>().size) // slab + overflow
        f.reset()
        assertEquals(0L, f.overflowBytes); assertFalse(o.isAlive); assertFalse(a.isAlive)
        assertTrue(sink.eventsOf<TraceEvent.Free>().any { it.storageId == o.id.value })
        // non-heap domains go to the platform and are treated as overflow too
        val p = f.allocate(64, MemoryDomain.HOST_OFFHEAP)
        assertEquals(ScopeKind.FORWARD, p.scope); assertTrue(f.liveBytes >= 64)
        f.reset(); assertFalse(p.isAlive)
    }

    @Test
    fun retainCopiesAStepResultOutOfTheForwardScope() {
        val f = ForwardScope(64)
        val act = f.allocateFloats(4)
        act.floats!![act.arrayOffset + 2] = 7f
        val kept = f.retain(act, origin = TensorId.parse("model.logits"))
        assertEquals(ScopeKind.AMBIENT, kept.scope); assertEquals(7f, kept.floats!![kept.arrayOffset + 2]); assertEquals("model.logits", kept.debugOrigin!!.canonical)
        f.reset()
        assertFalse(act.isAlive); assertTrue(kept.isAlive)
        val model = ModelScope()
        val inModel = f.retain(f.allocateFloats(2), to = model)
        assertEquals(ScopeKind.MODEL, inModel.scope)
        assertFailsWith<StorageClosedException> { f.retain(act) } // already reset
    }

    @Test
    fun modelScopeTracksAndClosesEverything() {
        val sink = RecordingTraceSink()
        val m = ModelScope(sink, "llama")
        val w = m.allocateFloats(16, TensorId.parse("model.embed_tokens.weight"))
        val o = m.allocate(32, MemoryDomain.HOST_OFFHEAP)
        val adopted = m.adopt(Storage.Heap.bytes(8))
        assertEquals(ScopeKind.MODEL, w.scope); assertEquals(ScopeKind.MODEL, o.scope); assertEquals(3, m.storageCount)
        assertEquals(64L + 32L + 8L, m.liveBytes)
        m.close(); m.close()
        assertTrue(m.isClosed); assertEquals(0L, m.liveBytes)
        assertFalse(w.isAlive); assertFalse(o.isAlive); assertFalse(adopted.isAlive)
        assertFailsWith<IllegalStateException> { m.allocateFloats(1) }
        // two of the three emit Free: the adopted storage was created with the default no-op sink
        assertEquals(2, sink.eventsOf<TraceEvent.Free>().size)
    }

    @Test
    fun scopesSeparateWeightsFromActivations() {
        // the two historical arena failures: weights and activations must not share a lifetime
        val model = ModelScope(); val fwd = ForwardScope(256)
        val weights = model.allocateFloats(64)
        repeat(10) { fwd.allocateFloats(128); fwd.allocateFloats(128); fwd.reset() }
        assertTrue(weights.isAlive); assertEquals(256, fwd.peakFloats); assertEquals(10L, fwd.steps)
        fwd.close(); assertTrue(weights.isAlive)
        model.close(); assertFalse(weights.isAlive)
    }
}
