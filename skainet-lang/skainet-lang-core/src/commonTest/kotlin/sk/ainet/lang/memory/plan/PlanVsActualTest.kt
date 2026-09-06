package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.ModelScope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * SKEEP-003 §4.9 / PRD M1-F8: the plan is compared with what a run actually allocated, and a drift
 * beyond the tolerance fails — which is what keeps the planner honest as kernels change.
 */
class PlanVsActualTest {

    private val geometry = ModelGeometry(layers = 2, heads = 4, kvHeads = 2, headDim = 16, valueDim = 16, embeddingLength = 64, feedForwardLength = 128, vocabSize = 256)

    private fun planFor(weightBytes: Long, ctx: Int = 128): MemoryPlan {
        val f = Format(FP32, TensorEncoding.Q4_K)
        val elements = weightBytes / 144 * 256
        val w = PlanTensor("w", TensorId.parse("model.w"), f, elements, weightBytes)
        return MemoryPlans.plan(PlanInput("tiny", "llama", listOf(w), geometry, ctx))
    }

    /** A run that allocates what the plan said: weights + KV in the model scope, the slab in forward. */
    private fun runMatching(plan: MemoryPlan): RecordingTraceSink {
        val sink = RecordingTraceSink()
        val model = ModelScope(sink)
        model.allocate(plan.weightsBytes + plan.kvBytes, origin = TensorId.parse("model.w"))
        val fwd = ForwardScope((plan.forwardBytes / 4).toInt(), sink)
        repeat(3) { step ->
            fwd.allocateFloats(((plan.forwardBytes / 4) / 2).toInt(), TensorId.parse("model.act#step=$step"))
            fwd.reset()
        }
        fwd.close(); model.close()
        return sink
    }

    @Test
    fun aRunThatMatchesThePlanIsWithinTolerance() {
        val plan = planFor(144L * 40)
        val cmp = PlanVsActual.of(plan, runMatching(plan))
        assertTrue(cmp.withinTolerance, cmp.render())
        assertTrue(cmp.violations().isEmpty())
        cmp.check()                                    // does not throw
        val text = cmp.render()
        assertTrue(text.contains("weights (model scope)")); assertTrue(text.contains("forward slab"))
        assertTrue(text.contains("forward-scope allocations:"))
        assertFalse(text.contains("✘"))
    }

    @Test
    fun aRunThatOverAllocatesFailsTheCheckWithATable() {
        val plan = planFor(144L * 40)
        val sink = RecordingTraceSink()
        val model = ModelScope(sink)
        model.allocate((plan.weightsBytes + plan.kvBytes) * 2)     // twice what was planned
        val cmp = PlanVsActual.of(plan, sink)
        assertFalse(cmp.withinTolerance)
        assertEquals(listOf("weights (model scope)", "forward slab"), cmp.violations().map { it.section })
        val e = assertFailsWith<IllegalStateException> { cmp.check() }
        assertTrue(e.message!!.contains("drifted")); assertTrue(e.message!!.contains("✘"))
        assertTrue(e.message!!.contains("+100 %"), e.message)
        model.close()
    }

    @Test
    fun actualsAreReconstructedFromTheEventStream() {
        val plan = planFor(144L * 10)
        val sink = runMatching(plan)
        val actual = ActualMemory.from(sink)
        assertEquals(plan.weightsBytes + plan.kvBytes, actual.peakModelBytes)
        assertTrue(actual.peakForwardBytes > 0)
        assertEquals(actual.peakModelBytes + actual.peakForwardBytes, actual.peakTotalBytes)
        // the forward scope allocated its slab once; the per-step views are slices, not allocations
        assertEquals(1, actual.allocationsByScope[ScopeKind.FORWARD])
        assertEquals(1, actual.allocationsByScope[ScopeKind.MODEL])
        assertEquals(0L, actual.adapterBytes)
    }

    @Test
    fun adapterBytesAreAccountedAndRendered() {
        val plan = planFor(144L * 10)
        val sink = runMatching(plan)
        sink.emit(
            sk.ainet.lang.memory.trace.TraceEvent.AdapterInserted(
                "dequantize", Format(FP32, TensorEncoding.Q6_K), Format.dense(FP32), 96L * 1024 * 1024, TensorId.parse("model.layers[3].mlp.down_proj.weight"),
            ),
        )
        val cmp = PlanVsActual.of(plan, sink)
        assertEquals(96L * 1024 * 1024, cmp.actual.adapterBytes)
        assertTrue(cmp.render().contains("adapters: 96 MB"), cmp.render())
        assertTrue(cmp.render().contains("dequantize"))
    }

    @Test
    fun toleranceIsConfigurableAndDefaultsToTenPercent() {
        assertEquals(0.10, PlanVsActual.DEFAULT_TOLERANCE)
        val plan = planFor(144L * 40)
        val sink = RecordingTraceSink()
        val model = ModelScope(sink)
        model.allocate((plan.weightsBytes + plan.kvBytes) * 108 / 100)   // 8 % over
        val fwd = ForwardScope((plan.forwardBytes / 4).toInt(), sink); fwd.allocateFloats(1)
        assertTrue(PlanVsActual.of(plan, sink).lines.first().withinTolerance(0.10))
        assertFalse(PlanVsActual.of(plan, sink, tolerance = 0.05).lines.first().withinTolerance(0.05))
        fwd.close(); model.close()
    }
}
