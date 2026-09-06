package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.storage.DefaultKvCacheStore
import sk.ainet.lang.tensor.storage.KvCacheConfig
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1036 (M2-F5, M2-A4): a sliding-window KV cache hands attention the one or two runs the ring
 * physically holds, and the answer is the same as if the window had never wrapped.
 */
class WindowedKvTest {

    private val heads = 2
    private val headDim = 4

    private fun store(maxSeqLen: Int, sliding: Boolean) = DefaultKvCacheStore(
        KvCacheConfig(numLayers = 1, numHeads = heads, headDim = headDim, maxSeqLen = maxSeqLen),
        null,
        sliding,
    )

    /** Deterministic per-token K/V so a position's contents identify it. */
    private fun token(step: Int, offset: Float = 0f) =
        FloatArray(heads * headDim) { i -> (step * 100 + i).toFloat() * 0.01f + offset }

    private fun fill(s: DefaultKvCacheStore, steps: Int) {
        for (step in 0 until steps) s.appendToken(0, token(step), token(step, offset = 0.5f))
    }

    // --- the ring ------------------------------------------------------------------------------

    @Test
    fun aNonSlidingCacheStillRefusesToOverflow() {
        val s = store(maxSeqLen = 4, sliding = false)
        fill(s, 4)
        assertFailsWith<IllegalStateException> { s.appendToken(0, token(4), token(4)) }
        assertEquals(0, s.windowStart, "without the ring, position 0 is still held")
    }

    @Test
    fun aRingKeepsTheNewestPositionsAndForgetsTheRest() {
        val s = store(maxSeqLen = 4, sliding = true)
        fill(s, 6)
        assertEquals(6, s.currentSeqLen, "positions stay absolute")
        assertEquals(2, s.windowStart, "the two oldest were overwritten")
        assertFailsWith<IllegalArgumentException> { s.keyWindow(0, from = 1, to = 6) }
        assertFailsWith<IllegalArgumentException> { s.readKeys(0, 0, 6) }
    }

    @Test
    fun aWrappedWindowIsTwoRunsInPositionOrder() {
        val s = store(maxSeqLen = 4, sliding = true)
        fill(s, 6)   // slots hold positions 4,5,2,3 — the window 2..6 wraps
        val w = s.keyWindow(0, from = 2, to = 6)
        assertTrue(w.wrapped, "the window crosses the end of the ring")
        assertEquals(2, w.parts.size)
        assertEquals(4, w.length)
        assertEquals(listOf(2, 2), w.parts.map { it.shape[WindowedKV.POSITION_AXIS] })

        // oldest first: positions 2,3,4,5
        for ((index, step) in listOf(2, 3, 4, 5).withIndex()) {
            val expected = token(step)
            for (h in 0 until heads) for (d in 0 until headDim) {
                assertEquals(expected[h * headDim + d], w.get(h, index, d), "position $step ($h,$d)")
            }
        }
    }

    @Test
    fun anUnwrappedWindowIsASingleRun() {
        val s = store(maxSeqLen = 8, sliding = true)
        fill(s, 5)
        val w = s.keyWindow(0, from = 1, to = 5)
        assertFalse(w.wrapped)
        assertEquals(1, w.parts.size)
        assertEquals(4, w.length)
    }

    @Test
    fun readingAcrossTheWrapReturnsPositionsInOrder() {
        val s = store(maxSeqLen = 4, sliding = true)
        fill(s, 7)
        val flat = s.readKeys(0, 3, 7)
        // [heads, 4 positions, headDim], oldest first
        for ((index, step) in listOf(3, 4, 5, 6).withIndex()) {
            val expected = token(step)
            for (h in 0 until heads) for (d in 0 until headDim) {
                assertEquals(expected[h * headDim + d], flat[(h * 4 + index) * headDim + d], "position $step")
            }
        }
    }

    // --- the views are views -------------------------------------------------------------------

    @Test
    fun aWindowIsZeroCopyOverTheCachesOwnStorage() {
        val s = store(maxSeqLen = 4, sliding = true)
        fill(s, 6)
        val first = s.keyWindow(0, 2, 6)
        val second = s.keyWindow(0, 2, 6)
        assertEquals(first.head.storage.id, second.head.storage.id, "windows share the cache's storage, they do not copy it")
        assertEquals(first.head.storage.id, first.tail!!.storage.id, "and both halves are the same storage")

        // a later append is visible through a window taken before it
        val w = s.keyWindow(0, 3, 6)
        s.appendToken(0, token(99), token(99, 0.5f))
        assertEquals(s.keyWindow(0, 3, 7).length, w.length + 1)
    }

    // --- M2-A4: the wrapped window computes what an unwrapped one would ------------------------

    @Test
    fun wrappedAttentionMatchesANonRingRunOverTheSameWindow() {
        val windowLength = 4
        val ring = store(maxSeqLen = windowLength, sliding = true)
        fill(ring, 10)                                    // wrapped several times: window is 6..10

        // the same positions, in a cache that never wrapped
        val flat = store(maxSeqLen = 64, sliding = false)
        for (step in 6 until 10) flat.appendToken(0, token(step), token(step, offset = 0.5f))

        val query = FloatArray(heads * headDim) { i -> 0.1f * (i + 1) }
        val fromRing = FloatArray(heads * headDim)
        val fromFlat = FloatArray(heads * headDim)
        WindowedAttention.decodeStep(query, ring.keyWindow(0, 6, 10), ring.valueWindow(0, 6, 10), fromRing)
        WindowedAttention.decodeStep(query, flat.keyWindow(0, 0, 4), flat.valueWindow(0, 0, 4), fromFlat)

        assertTrue(ring.keyWindow(0, 6, 10).wrapped, "the ring's window must actually wrap for this to mean anything")
        assertContentEquals(fromFlat, fromRing, "a wrapped window must produce the same logits as a flat one")
    }

    @Test
    fun theGatherAdapterAgreesWithThePairPathAndSaysSo() {
        val ring = store(maxSeqLen = 4, sliding = true)
        fill(ring, 6)
        val keys = ring.keyWindow(0, 2, 6)
        val values = ring.valueWindow(0, 2, 6)
        val query = FloatArray(heads * headDim) { i -> 0.05f * (i + 3) }

        val direct = FloatArray(heads * headDim)
        WindowedAttention.decodeStep(query, keys, values, direct)

        val sink = RecordingTraceSink()
        val scope = ForwardScope(slabFloats = heads * 4 * headDim * 2 + 64, sink = sink, name = "attn")
        val gathered = FloatArray(heads * headDim)
        WindowedAttention.decodeStepGathered(query, keys, values, gathered, scope, sink)

        for (i in direct.indices) {
            assertTrue(abs(direct[i] - gathered[i]) < 1e-5f, "element $i: ${direct[i]} vs ${gathered[i]}")
        }
        val adapters = sink.eventsOf<TraceEvent.AdapterInserted>()
        assertEquals(2, adapters.size, "one visible adapter per gathered window (keys, values)")
        assertTrue(adapters.all { it.kind == "gather-kv-window" }, adapters.map { it.kind }.toString())
        assertTrue(adapters.all { it.bytes == heads.toLong() * 4 * headDim * 4 }, "the copy is priced in the trace")
        scope.close()
    }

    @Test
    fun theZeroCopyPathAllocatesNothingPerToken() {
        val sink = RecordingTraceSink()
        val s = store(maxSeqLen = 4, sliding = true)
        fill(s, 8)
        val query = FloatArray(heads * headDim) { 0.25f }
        val out = FloatArray(heads * headDim)
        sink.clear()
        repeat(16) {
            WindowedAttention.decodeStep(query, s.keyWindow(0, s.windowStart, s.currentSeqLen), s.valueWindow(0, s.windowStart, s.currentSeqLen), out)
        }
        assertEquals(0, sink.eventsOf<TraceEvent.Allocation>().size, "windows and the pair kernel allocate nothing")
        assertEquals(0, sink.eventsOf<TraceEvent.AdapterInserted>().size, "and insert no adapters")
    }

    @Test
    fun everyStoreCanProduceAWindowEvenWithoutARing() {
        // the interface default copies once; it must still describe the same positions
        val turbo = sk.ainet.lang.tensor.storage.KvCacheStore.turboQuant(
            numLayers = 1, numHeads = heads, headDim = headDim, maxSeqLen = 8,
        )
        for (step in 0 until 3) turbo.appendToken(0, token(step), token(step, 0.5f))
        val w = turbo.keyWindow(0, 0, 3)
        assertFalse(w.wrapped, "a compressed store hands back one run")
        assertEquals(3, w.length)
        assertEquals(heads, w.heads)
        assertEquals(headDim, w.headDim)
    }
}
