package sk.ainet.exec.harness

import sk.ainet.lang.memory.trace.Counters
import sk.ainet.lang.memory.trace.GenerationMetrics
import sk.ainet.lang.memory.trace.PerfettoTraceExporter
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * #1035 (SKEEP-003 §4.9) end to end: a traced generation loop reports on itself.
 *
 * [GenerationMetricsTest][sk.ainet.lang.memory.trace] pins the arithmetic on a synthetic stream;
 * this pins that a *real* loop — prompt pass, decode steps, sampling, dispatch through
 * `KernelDispatch` — opens the spans the reader needs and produces numbers that make sense,
 * including the effective bandwidth, on every target the harness runs on.
 */
class GenerationMetricsHarnessTest {

    private val promptTokens = 3
    private val steps = 6

    private fun run(): DecodeHarness = DecodeHarness().also {
        it.prefill(promptTokens)
        it.decode(steps)
    }

    @Test
    fun theLoopReportsItsPhases() {
        val h = run()
        val m = h.metrics()
        assertEquals(promptTokens, m.prefillTokens, "the prefill span carries the prompt length")
        assertEquals(steps, m.decodeSteps)
        assertNotNull(m.timeToFirstTokenNanos, "TTFT spans the prompt pass and the first token")
        assertTrue(m.timeToFirstTokenNanos!! >= 0)
        assertTrue(m.sampleNanos >= 0)
        h.close()
    }

    @Test
    fun everyWeightShowsUpInThePerModuleBreakdown() {
        val h = run()
        val m = h.metrics()
        // one module span per weight, entered once per prompt token and once per decode step
        assertEquals(2 * DecodeHarness().layers, m.modules.size, "two projections per layer")
        for (module in m.modules) {
            assertEquals(promptTokens + steps, module.calls, "${module.path}: called once per token")
            assertTrue(module.nanos >= 0)
        }
        assertTrue(m.modules.any { it.path.endsWith("attn") }, "attention modules: ${m.modules.map { it.path }}")
        assertTrue(m.modules.any { it.path.endsWith("mlp") })
        h.close()
    }

    @Test
    fun effectiveBandwidthCountsTheBytesDecodeActuallyRead() {
        val h = run()
        val m = h.metrics(peakBytesPerSecond = 50L * 1024 * 1024 * 1024)
        assertTrue(m.kernelRunsDuringDecode > 0, "matmuls run inside the decode spans")
        assertEquals(steps * 2 * DecodeHarness().layers, m.kernelRunsDuringDecode, "one matmul per weight per step")
        assertTrue(m.bytesReadDuringDecode > 0, "a decode step reads its weights")
        // the prompt pass runs the same kernels: its bytes must not be counted as decode bandwidth
        val perStep = m.bytesReadDuringDecode / steps
        assertTrue(perStep > 0 && m.bytesReadDuringDecode == perStep * steps, "bytes read per step: $perStep")
        // rates are null on a clock too coarse to time these tiny steps, never infinite
        val bandwidth = m.effectiveBandwidthBytesPerSecond
        assertTrue(bandwidth == null || bandwidth > 0.0, "bandwidth: $bandwidth")
        assertTrue(m.bandwidthUtilization == null || m.bandwidthUtilization!! > 0.0)
        assertEquals(0, m.adapterCount, "a well-formed decode step needs no adapters")
        h.close()
    }

    @Test
    fun theMetricsReachThePerfettoTrace() {
        val h = run()
        GenerationMetrics.from(h.sink, peakBytesPerSecond = 50L * 1024 * 1024 * 1024).emitTo(h.sink)
        val json = PerfettoTraceExporter.export(h.sink)
        assertTrue(json.contains("\"name\":\"prefill\""), "the prompt pass is a span")
        assertTrue(json.contains("\"name\":\"decode#1\""), "decode steps are numbered")
        assertTrue(json.contains("\"name\":\"sample#1\""), "sampling is a span")
        assertTrue(json.contains(".attn#1\""), "module spans are labelled by module path")
        assertTrue(json.contains("\"ph\":\"C\",\"name\":\"${Counters.TIME_TO_FIRST_TOKEN}\""), "TTFT counter")
        h.close()
    }
}
