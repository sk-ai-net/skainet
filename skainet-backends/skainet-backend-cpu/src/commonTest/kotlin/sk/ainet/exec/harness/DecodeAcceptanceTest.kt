package sk.ainet.exec.harness

import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.plan.ActualMemory
import sk.ainet.lang.memory.plan.PlanVsActual
import sk.ainet.lang.memory.trace.PerfettoTraceExporter
import sk.ainet.lang.memory.trace.TraceEvent
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Milestone M1's acceptance criteria, asserted on every commit against the synthetic decode harness
 * (#1032 option (c)): the memory behaviour is checked here, where the memory code lives; the real
 * model's tok/s and TTFT belong to `skainet-decode` in SKaiNET-transformers.
 */
class DecodeAcceptanceTest {

    // Enough steps to prove the staircase is flat, few enough for Karma's 2 s per-test budget in a
    // browser (the reference kernel decodes every element, so wasm is the slowest target here).
    private val steps = 12

    @Test
    fun m1a1_memoryIsFlatAcrossDecodeSteps() {
        val h = DecodeHarness()
        h.decode(steps)
        // live bytes after the last step must equal live bytes after warm-up: the forward scope is
        // recycled, the model scope holds weights + KV and nothing else grows.
        val live = h.liveBytes()
        val model = live[ScopeKind.MODEL] ?: 0L
        val forward = live[ScopeKind.FORWARD] ?: 0L
        assertEquals(0L, forward, "the forward scope must be empty between steps")
        assertTrue(model > 0, "weights and the KV ring stay resident")

        // and the per-step resets all report the same live-bytes-before, i.e. a flat staircase
        val resets = h.sink.eventsOf<TraceEvent.ScopeReset>()
        assertEquals(steps, resets.size)
        val warmed = resets.drop(3).map { it.liveBytesBefore }.toSet()
        assertEquals(1, warmed.size, "per-step forward use must be identical after warm-up, saw $warmed")
        assertTrue(resets.all { it.liveBytesAfter == 0L })
        h.close()
    }

    @Test
    fun m1a3_noForwardScopeAllocationsPerStepAfterWarmUp() {
        val h = DecodeHarness()
        h.decode(12)
        // the slab is allocated once, before the first step; steps 5..20 must add nothing
        assertEquals(0, h.allocationsBetweenSteps(ScopeKind.FORWARD, fromStep = 4, toStep = 12), "steady-state decode must not allocate")
        assertEquals(0, h.allocationsBetweenSteps(ScopeKind.MODEL, fromStep = 1, toStep = 12), "weights and KV are allocated before decoding")
        h.close()
    }

    @Test
    fun m1a8_planMatchesWhatTheRunAllocated() {
        val h = DecodeHarness()
        h.decode(12)
        val cmp = PlanVsActual.of(h.plan(), h.sink)
        assertTrue(cmp.withinTolerance, "plan drifted from the run:\n" + cmp.render())
        val actual = ActualMemory.from(h.sink)
        assertTrue(actual.peakModelBytes > 0)
        assertEquals(0L, actual.adapterBytes, "a well-formed decode step needs no adapters")
        h.close()
    }

    @Test
    fun m1a7_theTraceHasOneTrackPerScopeAndAFlatLiveBytesCounter() {
        val h = DecodeHarness()
        h.decode(10)
        val json = PerfettoTraceExporter.export(h.sink, processName = "skainet-decode-harness")
        assertTrue(json.contains("model scope") && json.contains("forward scope"), "one track per scope")
        assertTrue(json.contains("\"ph\":\"X\",\"name\":\"matmul\""), "kernel spans")
        assertTrue(json.contains("model.layers[0].attn.q_proj.weight"), "spans labelled by TensorId")
        assertTrue(json.contains("\"ph\":\"C\",\"name\":\"live bytes\""), "a live-bytes counter track")
        assertTrue(json.contains("\"forward\":0"), "the counter returns to zero at every reset")
        h.close()
    }

    @Test
    fun theHarnessActuallyExercisesTheMemoryModel() {
        val h = DecodeHarness()
        h.decode(3)
        val kernels = h.sink.eventsOf<TraceEvent.KernelRun>()
        assertTrue(kernels.isNotEmpty(), "matmuls must go through KernelDispatch")
        assertTrue(kernels.all { it.op == "matmul" })
        assertTrue(h.sink.eventsOf<TraceEvent.Allocation>().any { it.scope == ScopeKind.MODEL })
        assertTrue(h.sink.eventsOf<TraceEvent.PhaseBegin>().count { it.phase == "decode" } == 3)
        h.close()
    }
}
