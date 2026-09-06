package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * #1143 / #1133: placement is a resolver decision — (what will be held × profile × platform) in,
 * an [sk.ainet.lang.memory.AllocationSpec] out. These tests pin the rules the old
 * `PlanTensor.allocation` hardcode ignored.
 */
class AllocationResolverTest {

    private val q4k = Format(FP32, TensorEncoding.Q4_K)

    private fun weight(
        elements: Long = 1L shl 20, // 1Mi elements: Q4_K packed ≈ 576 KiB, dense FP32 = 4 MiB
        form: WeightForm?,
    ) = PlanTensor(
        name = "blk.0.attn_q.weight",
        id = null,
        format = q4k,
        elementCount = elements,
        bytes = q4k.physicalBytes(elements)!!,
        form = form,
    )

    @Test
    fun mappedKeptAsStoredWeightIsServedFromTheFile() {
        val spec = AllocationResolver.resolve(
            weight(form = WeightForm(residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB,
            StorageCapabilities.FULL,
        )
        assertEquals(MemoryDomain.MMAP_FILE, spec.domain)
        assertEquals(ScopeKind.MODEL, spec.scope)
        assertFalse(spec.mutable)
        assertEquals(q4k, spec.format)
    }

    @Test
    fun unmappablePlatformFallsBackByTheProfileThreshold() {
        val spec = AllocationResolver.resolve(
            weight(form = WeightForm(residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB,
            StorageCapabilities(supportsMappedFiles = false),
        )
        // 576 KiB packed is over the 256 KiB off-heap threshold
        assertEquals(MemoryDomain.HOST_OFFHEAP, spec.domain)
        assertEquals(ScopeKind.MODEL, spec.scope)
    }

    @Test
    fun heapOnlyPlatformEndsOnTheHeapNoMatterTheSize() {
        val spec = AllocationResolver.resolve(
            weight(form = WeightForm(residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB,
            StorageCapabilities.HEAP_ONLY,
        )
        assertEquals(MemoryDomain.HOST_HEAP, spec.domain)
    }

    @Test
    fun dequantizedWeightIsNeverMapped() {
        val form = WeightForm(encoding = EncodingRequest.DequantizeTo(FP32), residency = WeightResidency.MAPPED)
        val spec = AllocationResolver.resolve(weight(form = form), PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL)
        assertNotEquals(MemoryDomain.MMAP_FILE, spec.domain)
        // the resident bytes are the dense bytes, and they price the domain decision
        assertEquals(Format.dense(FP32), spec.format)
        assertEquals(MemoryDomain.HOST_OFFHEAP, spec.domain) // 4 MiB dense is far over threshold
    }

    @Test
    fun kernelFeedOrderIsALoadTimeCopySoNotMapped() {
        val form = WeightForm(order = WeightByteOrder.KERNEL_FEED, residency = WeightResidency.MAPPED)
        val spec = AllocationResolver.resolve(weight(form = form), PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL)
        assertNotEquals(MemoryDomain.MMAP_FILE, spec.domain)
    }

    @Test
    fun smallHeapWeightStaysOnTheHeap() {
        val dense = Format.dense(FP32)
        val bias = PlanTensor("blk.0.attn_q.bias", null, dense, 768, 4L * 768, WeightForm())
        val spec = AllocationResolver.resolve(bias, PlannerProfile.DESKTOP, StorageCapabilities.FULL)
        assertEquals(MemoryDomain.HOST_HEAP, spec.domain)
        assertEquals(ScopeKind.MODEL, spec.scope)
    }

    @Test
    fun noFormMeansTheFileBytesByTheProfileRules() {
        val spec = AllocationResolver.resolve(weight(form = null), PlannerProfile.DESKTOP, StorageCapabilities.FULL)
        assertNotEquals(MemoryDomain.MMAP_FILE, spec.domain) // nothing asked for mapping
        assertEquals(MemoryDomain.HOST_OFFHEAP, spec.domain)
    }

    @Test
    fun transientAllocationsFollowScopeAndThreshold() {
        val dense = Format.dense(FP32)
        val small = AllocationResolver.resolveTransient(dense, 1024, PlannerProfile.DESKTOP, StorageCapabilities.FULL)
        assertEquals(MemoryDomain.HOST_HEAP, small.domain)
        assertEquals(ScopeKind.FORWARD, small.scope)
        assertTrue(small.mutable)

        val big = AllocationResolver.resolveTransient(
            dense, 1L shl 20, PlannerProfile.DESKTOP, StorageCapabilities.FULL, scope = ScopeKind.AMBIENT
        )
        assertEquals(MemoryDomain.HOST_OFFHEAP, big.domain)
        assertEquals(ScopeKind.AMBIENT, big.scope)
    }

    @Test
    fun explainSaysWhereAndWhy() {
        val mapped = AllocationResolver.explain(
            weight(form = WeightForm(residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL,
        )
        assertTrue("MMAP_FILE" in mapped && "file's bytes" in mapped, mapped)

        val noMmap = AllocationResolver.explain(
            weight(form = WeightForm(residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB, StorageCapabilities(supportsMappedFiles = false),
        )
        assertTrue("cannot map" in noMmap, noMmap)

        val dequant = AllocationResolver.explain(
            weight(form = WeightForm(encoding = EncodingRequest.DequantizeTo(FP32), residency = WeightResidency.MAPPED)),
            PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL,
        )
        assertTrue("re-encoded at load" in dequant, dequant)
    }

    @Test
    fun planTensorAllocationDelegatesToTheResolver() {
        val w = weight(form = WeightForm(residency = WeightResidency.MAPPED))
        assertEquals(
            AllocationResolver.resolve(w, PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL),
            w.allocation(PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL),
        )
    }
}
