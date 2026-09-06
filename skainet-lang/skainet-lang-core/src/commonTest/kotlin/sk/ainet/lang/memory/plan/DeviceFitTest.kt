package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1038 (SKEEP-002, M2-A5): a phone has **two** memory pools, and the plan has to be checked
 * against both.
 *
 * The managed heap is hard-capped per app and holds every Kotlin array; physical RAM holds
 * everything including mapped pages. A 600 MB Q4 model is impossible under a 512 MB cap and
 * unremarkable when its weights are mapped — one total cannot express that, so this asserts the
 * two-pool arithmetic and the advice it produces.
 */
class DeviceFitTest {

    private val mb = 1024L * 1024L

    /** A Llama-1B-shaped Q4_K model: ~640 MB of weights. */
    private fun plan(ctx: Int = 2048, weightsMb: Long = 640): MemoryPlan {
        val f = Format(FP32, TensorEncoding.Q4_K)
        val elementsPerTensor = weightsMb * mb / 144 * 256   // Q4_K: 144 bytes per 256 elements
        val tensors = listOf(PlanTensor("model.weight", null, f, elementsPerTensor, weightsMb * mb))
        val geometry = ModelGeometry(
            layers = 16, heads = 32, kvHeads = 8, headDim = 64,
            embeddingLength = 2048, feedForwardLength = 5632, vocabSize = 32000,
        )
        return MemoryPlans.plan(PlanInput("llama-1b", "llama", tensors, geometry, ctx))
    }

    /** A 2 GB phone: 512 MB largeHeap cap, ~900 MB RAM free, OS kills below 180 MB. */
    private fun phone(heapMaxMb: Long = 512, heapUsedMb: Long = 40, availMb: Long = 900) = DeviceMemory(
        totalRamBytes = 2048 * mb,
        availableRamBytes = availMb * mb,
        heapMaxBytes = heapMaxMb * mb,
        heapUsedBytes = heapUsedMb * mb,
        lowMemoryThresholdBytes = 180 * mb,
    )

    @Test
    fun heapWeightsDoNotFitUnderTheArtCapButMappedOnesDo() {
        val p = plan()
        val onHeap = p.fitOn(phone(), weightsMapped = false)
        assertFalse(onHeap.fits, "640 MB of weights cannot live under a 512 MB cap:\n${onHeap.render()}")
        assertEquals("managed heap", onHeap.blockingPool)
        assertTrue(onHeap.heap.neededBytes >= p.weightsBytes, "unmapped weights are charged to the heap")

        val mapped = p.fitOn(phone(), weightsMapped = true)
        assertTrue(mapped.heap.fits, "mapped weights never touch the managed heap:\n${mapped.render()}")
        assertEquals(p.kvBytes + p.forwardBytes + p.headroomBytes, mapped.heap.neededBytes)
    }

    @Test
    fun mappedPagesStillNeedPhysicalRam() {
        val p = plan()
        // plenty of heap, but the device has almost no free RAM: mapping is not free memory
        val squeezed = p.fitOn(phone(availMb = 300), weightsMapped = true)
        assertFalse(squeezed.fits, squeezed.render())
        assertEquals("device RAM", squeezed.blockingPool)
        assertEquals(p.totalBytes, squeezed.ram.neededBytes, "RAM carries everything, mapped included")
    }

    @Test
    fun theRamBudgetKeepsTheDeviceAboveItsOwnLowMemoryThreshold() {
        val p = plan(weightsMb = 100)
        val device = phone(availMb = 600)
        val fit = p.fitOn(device, weightsMapped = true)
        assertEquals((600 - 180) * mb, fit.ram.budgetBytes, "available minus the OS's own kill threshold")

        // a device that reports no threshold still keeps a floor
        val noThreshold = device.copy(lowMemoryThresholdBytes = 0)
        assertEquals(600 * mb - DeviceMemory.RAM_RESERVE_FLOOR, noThreshold.usableRamBytes)
    }

    @Test
    fun theFirstAdviceIsToMapTheWeights() {
        val fit = plan().fitOn(phone(), weightsMapped = false)
        val first = fit.suggestions.first()
        assertTrue(first.text.contains("MAPPED"), "expected mapping advice first, got '${first.text}'")
        assertEquals(fit.plan.weightsBytes, first.savesBytes, "it saves exactly the weights")
        assertTrue(fit.suggestions.size > 1, "and the plan's own suggestions follow: ${fit.suggestions.map { it.text }}")
    }

    @Test
    fun aPlanThatFitsHasNoBlockingPoolAndNoAdvice() {
        val fit = plan(ctx = 512, weightsMb = 120).fitOn(phone(), weightsMapped = true)
        assertTrue(fit.fits, fit.render())
        assertNull(fit.blockingPool)
        assertTrue(fit.suggestions.isEmpty())
        assertTrue(fit.heap.headroomBytes > 0 && fit.ram.headroomBytes > 0)
    }

    @Test
    fun theRenderedTableNamesThePoolAndTheShortfall() {
        val text = plan().fitOn(phone(), weightsMapped = false).render()
        assertTrue(text.contains("weights on the heap"), text)
        assertTrue(text.contains("managed heap"), text)
        assertTrue(text.contains("device RAM"), text)
        assertTrue(text.contains("short by"), text)
        assertTrue(plan(ctx = 512, weightsMb = 120).fitOn(phone(), weightsMapped = true).render().contains("weights mapped"))
    }

    @Test
    fun aDeviceUnderPressureSaysSo() {
        val fit = plan(weightsMb = 100).fitOn(phone().copy(lowMemory = true), weightsMapped = true)
        assertTrue(fit.render().contains("low memory"), fit.render())
    }

    @Test
    fun heapUseCountsAgainstTheCap() {
        val p = plan(weightsMb = 100)
        val fresh = p.fitOn(phone(heapUsedMb = 0), weightsMapped = false)
        val busy = p.fitOn(phone(heapUsedMb = 400), weightsMapped = false)
        assertEquals(fresh.heap.neededBytes, busy.heap.neededBytes)
        assertTrue(fresh.heap.budgetBytes > busy.heap.budgetBytes, "an app already holding 400 MB has less room")
    }
}
