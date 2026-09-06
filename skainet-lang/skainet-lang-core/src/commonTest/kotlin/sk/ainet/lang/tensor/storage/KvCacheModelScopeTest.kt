package sk.ainet.lang.tensor.storage

import sk.ainet.lang.memory.ModelScope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.plan.ActualMemory
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * SKEEP-003 §4.5 / PRD M1-F2: the KV cache preallocates its whole ring in `Scope.Model`, so the
 * bytes are tracked, traced and released with the model — and the memory plan's KV line can be
 * checked against what the store actually took (#1074).
 */
class KvCacheModelScopeTest {

    private val config = KvCacheConfig(numLayers = 4, numHeads = 2, headDim = 8, maxSeqLen = 16)

    private fun expectedBytes(): Long = 4L * 2 * (2 * 16 * 8) * 4   // layers × (K+V) × heads·seq·dim × 4 B

    @Test
    fun withoutAScopeTheStoreBehavesExactlyAsBefore() {
        val store = DefaultKvCacheStore(config)
        assertEquals(expectedBytes(), store.preallocatedBytes)
        store.appendToken(0, FloatArray(16) { 1f }, FloatArray(16) { 2f })
        for (l in 1 until 4) store.appendToken(l, FloatArray(16) { 1f }, FloatArray(16) { 2f })
        assertEquals(1, store.currentSeqLen)
        assertEquals(1f, store.readKeys(0, 0, 1)[0])
    }

    @Test
    fun withAModelScopeTheRingIsAllocatedTracedAndFreedWithTheModel() {
        val sink = RecordingTraceSink()
        val model = ModelScope(sink, "llama")
        val store = DefaultKvCacheStore(config, model)

        // one allocation per layer per side, all in MODEL scope, summing to the ring size
        val allocations = sink.eventsOf<TraceEvent.Allocation>()
        assertEquals(8, allocations.size, "4 layers × K and V")
        assertTrue(allocations.all { it.scope == ScopeKind.MODEL })
        assertEquals(expectedBytes(), allocations.sumOf { it.bytes })
        assertEquals(expectedBytes(), store.preallocatedBytes)
        assertEquals(expectedBytes(), model.liveBytes)
        // ids name the cache, not an anonymous buffer
        assertTrue(allocations.any { it.origin?.canonical == "kv.layers[0].k" }, allocations.map { it.origin?.canonical }.toString())
        assertTrue(allocations.any { it.origin?.canonical == "kv.layers[3].v" })

        // the cache still works
        for (l in 0 until 4) store.appendToken(l, FloatArray(16) { (l + 1).toFloat() }, FloatArray(16) { -1f })
        assertEquals(1, store.currentSeqLen)
        assertEquals(3f, store.readKeys(2, 0, 1)[0])

        // steady state: appending tokens allocates nothing more
        val before = sink.eventsOf<TraceEvent.Allocation>().size
        for (step in 1 until 5) for (l in 0 until 4) store.appendToken(l, FloatArray(16) { 1f }, FloatArray(16) { 1f })
        assertEquals(before, sink.eventsOf<TraceEvent.Allocation>().size, "appending must not allocate")
        assertEquals(5, store.currentSeqLen)

        // closing the model frees the ring
        model.close()
        assertEquals(0L, model.liveBytes)
        assertEquals(expectedBytes(), sink.eventsOf<TraceEvent.Free>().sumOf { it.bytes })
    }

    @Test
    fun theRingShowsUpInPlanVsActualAsModelScopeBytes() {
        val sink = RecordingTraceSink()
        val model = ModelScope(sink)
        DefaultKvCacheStore(config, model)
        val actual = ActualMemory.from(sink)
        assertEquals(expectedBytes(), actual.peakModelBytes)
        assertEquals(0L, actual.peakForwardBytes, "a KV ring is model-lifetime, never forward")
        assertEquals(8, actual.allocationsByScope[ScopeKind.MODEL])
        model.close()
        assertFalse(model.liveBytes > 0)
    }
}
