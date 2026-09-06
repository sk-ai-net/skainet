package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1035 (SKEEP-003 §4.9): the generation-loop metrics.
 *
 * The event stream is hand-built with explicit timestamps, so the arithmetic — TTFT, tok/s,
 * effective bandwidth, the per-module breakdown — is asserted exactly and identically on every
 * target, instead of depending on how coarse the platform clock happens to be.
 */
class GenerationMetricsTest {

    private val ms = 1_000_000L

    /** prefill(8 tokens, 40 ms) → 3 decode steps of 10 ms, each with a 2 ms sample. */
    private fun run(): List<TraceEvent> {
        val events = ArrayList<TraceEvent>()
        var t = 100L * ms
        events += TraceEvent.PhaseBegin(Phases.PREFILL, attributes = mapOf(Phases.ATTR_TOKENS to "8"), timeNanos = t)
        t += 40 * ms
        events += TraceEvent.PhaseEnd(Phases.PREFILL, durationNanos = 40 * ms, timeNanos = t)
        for (step in 1..3) {
            val begin = t
            events += TraceEvent.PhaseBegin(Phases.DECODE, step, timeNanos = begin)
            events += TraceEvent.PhaseBegin("model.layers[0].attn", step, mapOf(Phases.ATTR_KIND to Phases.KIND_MODULE), begin)
            events += TraceEvent.KernelRun("matmul", "reference", listOf(TensorId(listOf("model"), "w")), null, bytesRead = 1_000_000, bytesWritten = 4_000, durationNanos = 3 * ms, timeNanos = begin + 3 * ms)
            events += TraceEvent.PhaseEnd("model.layers[0].attn", step, durationNanos = 4 * ms, timeNanos = begin + 4 * ms)
            events += TraceEvent.PhaseBegin("model.layers[0].mlp", step, mapOf(Phases.ATTR_KIND to Phases.KIND_MODULE), begin + 4 * ms)
            events += TraceEvent.KernelRun("matmul", "reference", emptyList(), null, bytesRead = 2_000_000, bytesWritten = 4_000, durationNanos = 5 * ms, timeNanos = begin + 9 * ms)
            events += TraceEvent.PhaseEnd("model.layers[0].mlp", step, durationNanos = 6 * ms, timeNanos = begin + 10 * ms)
            events += TraceEvent.Counter(Counters.PAGE_FAULTS, value = 100L + step, unit = "faults", timeNanos = begin + 10 * ms)
            if (step == 2) {
                events += TraceEvent.AdapterInserted("dequantize", Format(FP32, TensorEncoding.Q8_0), Format.dense(FP32), bytes = 8_000, scope = ScopeKind.FORWARD, timeNanos = begin + 5 * ms)
            }
            t = begin + 10 * ms
            events += TraceEvent.PhaseEnd(Phases.DECODE, step, durationNanos = 10 * ms, timeNanos = t)
            events += TraceEvent.PhaseBegin(Phases.SAMPLE, step, timeNanos = t)
            t += 2 * ms
            events += TraceEvent.PhaseEnd(Phases.SAMPLE, step, durationNanos = 2 * ms, timeNanos = t)
        }
        return events
    }

    @Test
    fun theRatesComeOutOfTheSpans() {
        val m = GenerationMetrics.from(run())
        assertEquals(8, m.prefillTokens)
        assertEquals(40 * ms, m.prefillNanos)
        assertEquals(200.0, m.prefillTokensPerSecond, "8 tokens in 40 ms")
        assertEquals(3, m.decodeSteps)
        assertEquals(30 * ms, m.decodeNanos)
        assertEquals(100.0, m.decodeTokensPerSecond, "3 steps in 30 ms")
        assertEquals(10 * ms, m.nanosPerDecodeStep)
        assertEquals(6 * ms, m.sampleNanos)
    }

    @Test
    fun timeToFirstTokenSpansPrefillPlusTheFirstStepAndItsSample() {
        val m = GenerationMetrics.from(run())
        assertEquals(52 * ms, m.timeToFirstTokenNanos, "40 ms prefill + 10 ms decode + 2 ms sample")
    }

    @Test
    fun timeToFirstTokenFallsBackToTheFirstDecodeStepWhenThereIsNoPrompt() {
        val events = listOf(
            TraceEvent.PhaseBegin(Phases.DECODE, 1, timeNanos = 0L),
            TraceEvent.PhaseEnd(Phases.DECODE, 1, durationNanos = 7 * ms, timeNanos = 7 * ms),
        )
        val m = GenerationMetrics.from(events)
        assertEquals(7 * ms, m.timeToFirstTokenNanos)
        assertEquals(1, m.decodeSteps)
    }

    @Test
    fun effectiveBandwidthIsBytesReadOverDecodeTime() {
        val m = GenerationMetrics.from(run(), peakBytesPerSecond = 1_000_000_000L)
        assertEquals(9_000_000L, m.bytesReadDuringDecode, "3 steps x 3 MB")
        assertEquals(6, m.kernelRunsDuringDecode)
        assertEquals(24 * ms, m.kernelNanosDuringDecode)
        assertEquals(300_000_000.0, m.effectiveBandwidthBytesPerSecond, "9 MB in 30 ms")
        assertEquals(0.3, m.bandwidthUtilization!!, 1e-9, "30 % of a 1 GB/s device")
        assertEquals(0.8, m.kernelShareOfDecode!!, 1e-9)
    }

    @Test
    fun kernelsOutsideDecodeDoNotCountTowardsBandwidth() {
        // the prefill's kernels read far more than decode's — counting them would flatter the number
        val withPrefillKernel = run().toMutableList()
        withPrefillKernel.add(2, TraceEvent.KernelRun("matmul", "reference", bytesRead = 500_000_000, durationNanos = ms, timeNanos = 110L * ms))
        val m = GenerationMetrics.from(withPrefillKernel)
        assertEquals(9_000_000L, m.bytesReadDuringDecode)
    }

    @Test
    fun adaptersAndPageFaultsAreAttributedToTheDecodeWindow() {
        val m = GenerationMetrics.from(run())
        assertEquals(1, m.adapterCount)
        assertEquals(8_000L, m.adapterBytes)
        assertEquals(8_000.0 / 9_000_000.0, m.adapterShareOfBytesRead!!, 1e-12)
        assertEquals(2L, m.pageFaultsDuringDecode, "counter went 101 → 103 across the decode steps")
        assertEquals(2.0 * 1_000_000_000.0 / (30 * ms), m.pageFaultsPerSecond!!, 1e-9)
    }

    @Test
    fun theModuleBreakdownIsOrderedByCost() {
        val m = GenerationMetrics.from(run())
        assertEquals(2, m.modules.size)
        assertEquals("model.layers[0].mlp", m.modules[0].path, "the expensive module comes first")
        assertEquals(18 * ms, m.modules[0].nanos)
        assertEquals(3, m.modules[0].calls)
        assertEquals(6 * ms, m.modules[0].averageNanos)
        assertEquals("model.layers[0].attn", m.modules[1].path)
        assertEquals(12 * ms, m.modules[1].nanos)
    }

    @Test
    fun anEmptyStreamHasNoRatesRatherThanInfiniteOnes() {
        val m = GenerationMetrics.from(emptyList())
        assertEquals(0, m.decodeSteps)
        assertNull(m.decodeTokensPerSecond)
        assertNull(m.effectiveBandwidthBytesPerSecond)
        assertNull(m.bandwidthUtilization)
        assertNull(m.timeToFirstTokenNanos)
        assertNull(m.pageFaultsPerSecond)
        assertNull(m.adapterShareOfBytesRead)
        assertTrue(m.render().contains("decode         0 steps"))
    }

    @Test
    fun theHelpersProduceTheSpansTheReaderExpects() {
        val sink = RecordingTraceSink()
        sink.prefill(tokens = 5) { }
        for (step in 1..2) {
            sink.decodeStep(step) {
                sink.module("model.layers[0].attn", step) { }
                sink.counter(Counters.PAGE_FAULTS, 7L + step, unit = "faults")
            }
            sink.sample(step) { }
        }
        val m = GenerationMetrics.from(sink)
        assertEquals(5, m.prefillTokens)
        assertEquals(2, m.decodeSteps)
        assertEquals(1, m.modules.size)
        assertEquals(2, m.modules[0].calls)
        assertEquals(1L, m.pageFaultsDuringDecode)
        assertTrue(m.timeToFirstTokenNanos != null && m.timeToFirstTokenNanos!! >= 0)
    }

    @Test
    fun aModuleSpanCanBeNamedAfterATensorId() {
        val sink = RecordingTraceSink()
        sink.decodeStep(1) {
            sink.module(TensorId(listOf("model", "layers[3]", "attn"), "q_proj.weight")) { }
        }
        assertEquals("model.layers[3].attn", GenerationMetrics.from(sink).modules.single().path)
    }

    @Test
    fun derivedMetricsAreEmittedAsCountersAndReachThePerfettoTrace() {
        val sink = RecordingTraceSink()
        run().forEach { sink.emit(it) }
        GenerationMetrics.from(sink, peakBytesPerSecond = 1_000_000_000L).emitTo(sink)

        val counters = sink.eventsOf<TraceEvent.Counter>().associate { it.name to it.value }
        assertEquals(52_000L, counters[Counters.TIME_TO_FIRST_TOKEN], "microseconds")
        assertEquals(200L, counters[Counters.PREFILL_TOKENS_PER_SECOND])
        assertEquals(100L, counters[Counters.DECODE_TOKENS_PER_SECOND])
        assertEquals(300_000_000L, counters[Counters.EFFECTIVE_BANDWIDTH])
        assertEquals(30L, counters[Counters.BANDWIDTH_UTILIZATION], "percent")

        val json = PerfettoTraceExporter.export(sink)
        for (name in listOf(Counters.TIME_TO_FIRST_TOKEN, Counters.DECODE_TOKENS_PER_SECOND, Counters.EFFECTIVE_BANDWIDTH, Counters.BANDWIDTH_UTILIZATION)) {
            assertTrue(json.contains("\"ph\":\"C\",\"name\":\"$name\""), "counter track '$name' missing from the trace")
        }
        assertTrue(json.contains("\"name\":\"prefill\""), "the prefill span")
        assertTrue(json.contains("\"name\":\"decode#1\""), "decode steps are numbered")
        assertTrue(json.contains("\"name\":\"sample#1\""), "the sample span")
        assertTrue(json.contains("\"name\":\"model.layers[0].mlp#1\""), "module spans")
    }
}
