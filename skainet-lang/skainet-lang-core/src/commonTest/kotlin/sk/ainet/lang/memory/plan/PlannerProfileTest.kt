package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1039 (M2-F6, decision #11): the planner's device profiles — one test per rule, because each of
 * them is a number someone will otherwise "improve" by feel.
 */
class PlannerProfileTest {

    private val mb = 1024L * 1024L

    private fun input(ctx: Int = 2048, weightsMb: Long = 600, prefillChunk: Int = 64): PlanInput {
        val f = Format(FP32, TensorEncoding.Q4_K)
        val elements = weightsMb * mb / 144 * 256
        return PlanInput(
            modelName = "llama-1b",
            architecture = "llama",
            weights = listOf(PlanTensor("model.weight", null, f, elements, weightsMb * mb)),
            geometry = ModelGeometry(
                layers = 16, heads = 32, kvHeads = 8, headDim = 64,
                embeddingLength = 2048, feedForwardLength = 5632, vocabSize = 32000,
            ),
            ctx = ctx,
            prefillChunk = prefillChunk,
            kvMode = KvCacheMode.FP32,
        )
    }

    // --- budget ------------------------------------------------------------------------------

    @Test
    fun theBudgetIsAvailableMinusTheProfilesReserve() {
        val available = 2048 * mb
        assertEquals(
            available - Budget.RESERVE_ANDROID_JVM,
            PlannerProfile.MOBILE_2GB.budget(available).bytes,
            "mobile reserves 700 MB for the OS and the app (decision #11)",
        )
        assertEquals(
            available - Budget.RESERVE_NATIVE,
            PlannerProfile.NATIVE.budget(available).bytes,
            "Kotlin/Native reserves 300 MB",
        )
        assertEquals(0L, PlannerProfile.MOBILE_2GB.budget(100 * mb).bytes, "never negative")
        assertTrue(PlannerProfile.MOBILE_2GB.budget(available).description.contains("mobile-2gb"))
    }

    // --- the profile's own defaults -----------------------------------------------------------

    @Test
    fun theProfilesDefaultsAreTheOnesDecisionElevenNames() {
        val m = PlannerProfile.MOBILE_2GB
        assertEquals(256, m.prefillChunk, "prefill chunked at 256 tokens")
        assertEquals(256L * 1024, m.offHeapThresholdBytes, "heap/off-heap threshold is 256 KB")
        assertEquals(0.80, m.kvAutoQuantizeAbove, "KV auto-quantizes over 80 % of the budget")
        assertEquals(0.05, m.dequantWarnFraction, "dispatcher dequant warns over 5 %")
        assertTrue(m.weightsMapped, "on a phone the weights are mapped")
        assertTrue(m.strict, "and a missing kernel fails rather than costing several times the weight")

        val d = PlannerProfile.DESKTOP
        assertEquals(1.0, d.kvAutoQuantizeAbove, "a desktop never silently re-quantizes the cache")
        assertFalse(d.weightsMapped, "desktop staging is the heap — today's behaviour, unchanged")
    }

    @Test
    fun theOffHeapThresholdDecidesWhereAnAllocationGoes() {
        val p = PlannerProfile.MOBILE_2GB
        assertEquals(MemoryDomain.HOST_HEAP, p.domainFor(255L * 1024))
        assertEquals(MemoryDomain.HOST_OFFHEAP, p.domainFor(256L * 1024), "at the threshold, not over it")
        assertEquals(MemoryDomain.HOST_OFFHEAP, p.domainFor(64 * mb))
    }

    // --- the forward slab and the KV cache -----------------------------------------------------

    @Test
    fun theProfileImposesItsPrefillChunkOnTheInput() {
        val profiled = PlannerProfile.MOBILE_2GB.plan(input(prefillChunk = 8), availableBytes = 4096 * mb)
        assertEquals(256, profiled.plan.input.prefillChunk, "the profile's slab width wins over the caller's")
    }

    @Test
    fun kvIsQuantizedAutomaticallyOnlyWhenThePlanIsTight() {
        // the same model at the same context length, planned twice: only the budget differs.
        val model = input(ctx = 8192)

        // comfortable: 8 GB available — the cache stays as planned
        val roomy = PlannerProfile.MOBILE_2GB.plan(model, availableBytes = 8192 * mb)
        assertEquals(KvCacheMode.BF16, roomy.plan.input.kvMode, roomy.render())
        assertTrue(roomy.notes.none { it.contains("switched") }, roomy.notes.toString())

        // tight: a 2 GB device — the cache quantizes before anything else gives
        val tight = PlannerProfile.MOBILE_2GB.plan(model, availableBytes = 2048 * mb)
        assertEquals(KvCacheMode.TURBOQUANT_4, tight.plan.input.kvMode, tight.render())
        assertTrue(tight.notes.any { it.contains("KV cache switched") }, tight.notes.toString())
        assertTrue(
            tight.plan.kvBytes < roomy.plan.kvBytes,
            "quantizing must shrink the cache: ${tight.plan.kvBytes} vs ${roomy.plan.kvBytes}",
        )
        assertTrue(tight.plan.totalBytes < roomy.plan.totalBytes, "and the plan with it")
    }

    @Test
    fun theDesktopProfileLeavesTheCacheAlone() {
        val tight = PlannerProfile.DESKTOP.plan(input(ctx = 8192), availableBytes = 2048 * mb)
        assertEquals(KvCacheMode.BF16, tight.plan.input.kvMode, "no automatic re-quantization off-device")
        assertTrue(tight.notes.isEmpty())
    }

    // --- the fit check ------------------------------------------------------------------------

    @Test
    fun theFitCheckRefusesBeforeAnythingIsAllocated() {
        val profiled = PlannerProfile.MOBILE_2GB.plan(input(weightsMb = 1600), availableBytes = 2048 * mb)
        assertEquals(false, profiled.fits)
        val failure = assertFailsWith<IllegalStateException> { profiled.requireFits() }
        assertTrue(failure.message!!.contains("mobile-2gb"), failure.message!!)
        assertTrue(profiled.plan.suggestions().isNotEmpty(), "and it says what to do instead")
    }

    @Test
    fun theFitCheckCanRefuseAgainstARealDevicesTwoPools() {
        val phone = DeviceMemory(
            totalRamBytes = 2048 * mb, availableRamBytes = 900 * mb,
            heapMaxBytes = 512 * mb, heapUsedBytes = 40 * mb, lowMemoryThresholdBytes = 180 * mb,
        )
        val mobile = PlannerProfile.MOBILE_2GB.plan(input(weightsMb = 600, ctx = 512), availableBytes = 900 * mb)
        mobile.requireFits(phone)   // mapped weights: the heap only carries KV + forward + headroom

        // the same plan with heap staging is charged for the weights and cannot fit
        val onHeap = PlannerProfile.DESKTOP.plan(input(weightsMb = 600, ctx = 512), availableBytes = 900 * mb)
        val failure = assertFailsWith<IllegalStateException> { onHeap.requireFits(phone) }
        assertTrue(failure.message!!.contains("managed heap"), failure.message!!)
    }

    // --- dispatcher-inserted dequantization ----------------------------------------------------

    @Test
    fun dequantizationUnderTheLimitIsFine() {
        val verdict = PlannerProfile.MOBILE_2GB.checkDequant(0.04)
        assertEquals(DequantSeverity.OK, verdict.severity, verdict.message)
        verdict.requireAcceptable()
        assertEquals(DequantSeverity.OK, PlannerProfile.MOBILE_2GB.checkDequant(null).severity, "nothing read, nothing to judge")
    }

    @Test
    fun dequantizationOverTheLimitWarnsAndFailsUnderStrict() {
        // A desktop has the memory to absorb a widening, so it is told and carries on.
        val lenient = PlannerProfile.DESKTOP.checkDequant(0.31)
        assertEquals(DequantSeverity.WARN, lenient.severity)
        assertTrue(lenient.message.contains("31.0%"), lenient.message)
        assertTrue(lenient.message.contains("kernel for the on-disk format is missing"), lenient.message)
        lenient.requireAcceptable()   // a warning does not stop a desktop run

        // A 2 GB board does not, so the same share is an error there without asking for strict.
        val strict = PlannerProfile.MOBILE_2GB.checkDequant(0.31)
        assertEquals(DequantSeverity.ERROR, strict.severity, "MOBILE_2GB is strict by default")
        val failure = assertFailsWith<IllegalStateException> { strict.requireAcceptable() }
        assertTrue(failure.message!!.contains("over the 5.0%"), failure.message!!)

        // strict() stays available for turning a lenient profile into a failing one in CI.
        assertTrue(PlannerProfile.DESKTOP.strict().checkDequant(0.31).profile.name.contains("strict"))
        assertEquals(DequantSeverity.ERROR, PlannerProfile.DESKTOP.strict().checkDequant(0.31).severity)
    }

    // --- picking a profile ---------------------------------------------------------------------

    @Test
    fun aDevicePicksItsOwnProfile() {
        fun device(ramMb: Long) = DeviceMemory(
            totalRamBytes = ramMb * mb, availableRamBytes = ramMb * mb / 2,
            heapMaxBytes = 512 * mb,
        )
        assertEquals(PlannerProfile.MOBILE_2GB, PlannerProfile.forDevice(device(2048)))
        assertEquals(PlannerProfile.MOBILE_2GB, PlannerProfile.forDevice(device(3072)), "3 GB phones are still phones")
        assertEquals(PlannerProfile.DESKTOP, PlannerProfile.forDevice(device(16384)))
    }

    @Test
    fun theRenderedPlanNamesTheProfileAndItsDecisions() {
        val text = PlannerProfile.MOBILE_2GB.plan(input(ctx = 8192), availableBytes = 2048 * mb).render()
        assertTrue(text.contains("profile mobile-2gb"), text)
        assertTrue(text.contains("prefill 256"), text)
        assertTrue(text.contains("off-heap ≥ 256"), text)
        assertTrue(text.contains("KV auto-quantize over 80.0%"), text)
        assertTrue(text.contains("note: KV cache switched"), text)
        assertTrue(text.contains("note: weights are counted resident and mapped"), text)
    }

    @Test
    fun domainForSplitsExactlyAtTheOffHeapThreshold() {
        val p = PlannerProfile.DESKTOP
        assertEquals(sk.ainet.lang.tensor.storage.MemoryDomain.HOST_HEAP, p.domainFor(p.offHeapThresholdBytes - 1))
        assertEquals(sk.ainet.lang.tensor.storage.MemoryDomain.HOST_OFFHEAP, p.domainFor(p.offHeapThresholdBytes))
        assertEquals(sk.ainet.lang.tensor.storage.MemoryDomain.HOST_HEAP, p.domainFor(0))
    }
}
