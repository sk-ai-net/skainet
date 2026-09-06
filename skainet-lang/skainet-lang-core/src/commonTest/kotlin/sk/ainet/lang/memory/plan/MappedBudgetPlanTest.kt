package sk.ainet.lang.memory.plan

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * #1189's planner half: weights served from file-backed pages are charged against device RAM, not
 * the heap [Budget]. Pinned by the Pixel 8a measurement — a 1.0 GB mapped Q4_K_M model decoded
 * under a 256 MB ART cap with 566 KB of weight heap; a plan that says "does not fit" about that
 * run is wrong, and this test is what keeps it from saying so again.
 */
class MappedBudgetPlanTest {

    private val platform = StorageCapabilities.FULL

    private fun q4k(name: String, bytes: Long, form: WeightForm?) = PlanTensor(
        name = name,
        id = null,
        format = Format(FP32, TensorEncoding.Q4_K),
        elementCount = bytes / 144 * 256,
        bytes = bytes,
        form = form,
    )

    private fun input(weights: List<PlanTensor>) = PlanInput(
        modelName = "m", architecture = "llama", weights = weights, geometry = null, ctx = 512,
    )

    @Test
    fun mapped_weights_are_not_charged_against_the_heap_budget() {
        val gig = 1024L * 1024 * 1024
        val cap = 256L * 1024 * 1024
        val mappedForm = WeightForm(residency = WeightResidency.MAPPED)
        val plan = MemoryPlans.plan(input(listOf(q4k("w", gig, mappedForm))), Budget.of(cap), platform)

        assertEquals(gig, plan.weightsMappedBytes)
        assertEquals(0L, plan.weightsHeapBytes)
        assertEquals(plan.totalBytes - gig, plan.budgetedBytes)
        assertEquals(true, plan.fits, "1 GB mapped weights under a 256 MB cap FIT — measured, #1189:\n" + plan.render())
        assertTrue(plan.suggestions().isEmpty(), "a fitting plan suggests nothing")

        val render = plan.render()
        assertTrue("mapped" in render && "✔ fits" in render, render)
        assertTrue("total heap" in render, render)
    }

    @Test
    fun heap_staged_weights_still_count_and_still_overflow() {
        val gig = 1024L * 1024 * 1024
        val cap = 256L * 1024 * 1024
        val plan = MemoryPlans.plan(input(listOf(q4k("w", gig, form = null))), Budget.of(cap), platform)

        assertEquals(0L, plan.weightsMappedBytes)
        assertEquals(plan.totalBytes, plan.budgetedBytes)
        assertEquals(false, plan.fits)
        assertFalse(plan.suggestions().isEmpty(), "an over-budget plan must suggest a way out")
    }

    @Test
    fun encodings_nobody_serves_from_a_mapping_stay_heap_charged_even_under_MAPPED() {
        // was Q8_0 until #1192 made it servable — the pin moves to the ternary family
        val bytes = 100L * 1024 * 1024
        val mappedForm = WeightForm(residency = WeightResidency.MAPPED)
        // TQ2_0 needs a load-time repack — a copy cannot page from the file it no longer
        // matches — so it stays heap-charged until the repack cache lands (#1192 follow-up).
        val tq = PlanTensor("w", null, Format(FP32, TensorEncoding.TQ2_0), bytes * 4, bytes, mappedForm)
        val plan = MemoryPlans.plan(input(listOf(tq)), Budget.of(256L * 1024 * 1024), platform)

        assertEquals(0L, plan.weightsMappedBytes, "TQ2_0 has no mapped-servable representation")

        // and the resolver agrees — the plan and the load must tell the same story
        val spec = AllocationResolver.resolve(tq, PlannerProfile.MOBILE_2GB, platform)
        assertNotEquals(MemoryDomain.MMAP_FILE, spec.domain)
        assertTrue("no loader/kernel serves TQ2_0" in AllocationResolver.explain(tq, PlannerProfile.MOBILE_2GB, platform))
    }

    @Test
    fun a_mix_splits_into_a_mapped_and_a_heap_line() {
        val mappedForm = WeightForm(residency = WeightResidency.MAPPED)
        val big = q4k("big", 512L * 1024 * 1024, mappedForm)
        val tq = PlanTensor(
            "small", null, Format(FP32, TensorEncoding.TQ2_0),
            (10L * 1024 * 1024) * 4, 10L * 1024 * 1024, mappedForm,
        )
        val plan = MemoryPlans.plan(input(listOf(big, tq)), Budget.of(256L * 1024 * 1024), platform)

        assertEquals(512L * 1024 * 1024, plan.weightsMappedBytes)
        assertEquals(10L * 1024 * 1024, plan.weightsHeapBytes)
        val weightLines = plan.lines.filter { it.section == "weights" }
        assertEquals(2, weightLines.size)
        assertTrue(weightLines[0].mapped && !weightLines[1].mapped)
    }

    @Test
    fun without_mapping_support_everything_is_heap_charged() {
        val mappedForm = WeightForm(residency = WeightResidency.MAPPED)
        val plan = MemoryPlans.plan(
            input(listOf(q4k("w", 512L * 1024 * 1024, mappedForm))),
            Budget.of(256L * 1024 * 1024),
            StorageCapabilities.HEAP_ONLY,
        )
        assertEquals(0L, plan.weightsMappedBytes)
        assertEquals(false, plan.fits)
    }
}
