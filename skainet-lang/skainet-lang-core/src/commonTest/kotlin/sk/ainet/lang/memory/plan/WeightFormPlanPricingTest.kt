package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1116: a declared form changes the plan, so a dequantization is a line in a table rather than an
 * OOM at load.
 */
class WeightFormPlanPricingTest {

    private val elements = 1L shl 20   // 1 Mi weights, so the numbers are legible

    private fun q4kTensor(form: WeightForm? = null): PlanTensor {
        val format = Format(FP32, TensorEncoding.Q4_K)
        return PlanTensor(
            name = "blk.0.attn_q.weight",
            id = null,
            format = format,
            elementCount = elements,
            bytes = format.physicalBytes(elements)!!,
            form = form,
        )
    }

    private fun input(vararg weights: PlanTensor) = PlanInput(
        modelName = "synthetic", architecture = "llama", weights = weights.toList(),
        geometry = null, ctx = 2048,
    )

    @Test
    fun `a weight held as stored costs what the file holds`() {
        val stored = q4kTensor()
        assertEquals(stored.bytes, stored.residentBytes, "no form resolved: the file's bytes are the cost")

        val kept = q4kTensor(WeightForm(EncodingRequest.KeepAsStored))
        assertEquals(kept.bytes, kept.residentBytes, "KeepAsStored: same")
    }

    @Test
    fun `a dequantized weight costs its dense size rather than its packed one`() {
        val dequantized = q4kTensor(WeightForm(EncodingRequest.DequantizeTo(FP32)))
        assertEquals(elements * 4, dequantized.residentBytes, "FP32 is four bytes an element")
        assertTrue(
            dequantized.residentBytes > dequantized.bytes * 7,
            "Q4_K → FP32 is roughly 8×; got ${dequantized.bytes} → ${dequantized.residentBytes}",
        )
    }

    @Test
    fun `the plan totals the resolved size and reports what the conversion added`() {
        val plan = MemoryPlans.plan(input(q4kTensor(WeightForm(EncodingRequest.DequantizeTo(FP32)))))
        val stored = q4kTensor().bytes

        assertEquals(elements * 4, plan.weightsBytes, "the plan holds the dense weight")
        assertEquals(stored, plan.weightsAsStoredBytes, "and remembers what the file held")
        assertEquals(elements * 4 - stored, plan.formConversionBytes, "the difference is the conversion's price")
    }

    @Test
    fun `a plan that only fits as stored does not claim to fit once dequantized`() {
        // The failure mode #1116 exists to prevent: budget checked against the file's size, then
        // the load quadruples it.
        val stored = q4kTensor()
        val budget = Budget.of(stored.bytes + MemoryPlans.HEAP_HEADROOM_BYTES + 1)

        assertEquals(true, MemoryPlans.plan(input(stored), budget).fits, "as stored, it fits")

        val dequantized = MemoryPlans.plan(input(q4kTensor(WeightForm(EncodingRequest.DequantizeTo(FP32)))), budget)
        assertEquals(false, dequantized.fits, "dequantized, it does not — and the plan says so before the load")
    }

    @Test
    fun `the suggestion names the conversion since nobody asked for it`() {
        val budget = Budget.of(q4kTensor().bytes)
        val plan = MemoryPlans.plan(input(q4kTensor(WeightForm(EncodingRequest.DequantizeTo(FP32)))), budget)

        val suggestion = plan.suggestions().firstOrNull { it.text.contains("kernels for the stored encoding") }
        assertTrue(suggestion != null, "expected a suggestion about the conversion, got ${plan.suggestions()}")
        assertEquals(plan.formConversionBytes, suggestion.savesBytes, "it saves exactly what it costs")
    }

    @Test
    fun `the rendered table shows the conversion and hides it when there is none`() {
        val converted = MemoryPlans.plan(input(q4kTensor(WeightForm(EncodingRequest.DequantizeTo(FP32))))).render()
        assertTrue(converted.contains("re-encoded at load"), converted)

        val asStored = MemoryPlans.plan(input(q4kTensor())).render()
        assertTrue(!asStored.contains("re-encoded at load"), "an unconverted plan reads exactly as before:\n$asStored")
    }

    @Test
    fun `resolving an input prices what the target can actually feed`() {
        val stored = input(q4kTensor())

        val withKernels = stored.resolveWeightForms(PlannerProfile.DESKTOP, KernelCapabilities.EVERYTHING)
        assertEquals(
            MemoryPlans.plan(stored).weightsBytes, MemoryPlans.plan(withKernels).weightsBytes,
            "a target that can feed Q4_K holds Q4_K, and the plan is unchanged",
        )

        val withoutKernels = stored.resolveWeightForms(PlannerProfile.DESKTOP, KernelCapabilities.DENSE_ONLY)
        assertEquals(
            elements * 4, MemoryPlans.plan(withoutKernels).weightsBytes,
            "a target with only dense kernels holds FP32 — the same file, four times the memory",
        )
    }
}
