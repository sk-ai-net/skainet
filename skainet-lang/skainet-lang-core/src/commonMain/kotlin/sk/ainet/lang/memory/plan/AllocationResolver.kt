package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.AllocationSpec
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.PlatformStorage
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * What the running platform's storage can actually do — the third input of [AllocationResolver],
 * separated from [PlatformStorage] so a test can resolve for a platform it is not running on.
 */
public data class StorageCapabilities(
    val supportsMappedFiles: Boolean,
    val supportsOffHeap: Boolean = true,
    /**
     * Encodings whose bytes the runtime can actually *serve* from a file mapping — a tensor is
     * only truly mapped when a loader emits a file-backed representation for it AND a kernel can
     * read it there. Today (#921 dense F32, #1189 buffer-packed Q4_K/Q6_K) that is exactly
     * [MAPPED_SERVABLE_DEFAULT]; every other encoding under `WeightResidency.MAPPED` falls back
     * to heap staging, and a plan that assumed otherwise would under-count the heap it is about
     * to fill (the mirror image of the #1116 dequantize-surprise).
     */
    val mappedServableEncodings: Set<TensorEncoding> = MAPPED_SERVABLE_DEFAULT,
) {
    public companion object {
        /**
         * What the loaders can serve from mapped pages today: dense FP32 (#921) and the GGML
         * block formats (#1189 Q4_K/Q6_K, #1192 the rest). Ternary formats need a load-time
         * repack and stay heap until the repack cache lands.
         *
         * This module cannot depend on `skainet-backend-api` (that dependency runs the other
         * way, to avoid a cycle), so this constant can't be derived from
         * `KernelDispatch.mappedServableEncodings()` directly (#1193). It is instead
         * cross-checked there: `KernelSupportMatrixTest.generate_and_gate_support_matrix()`
         * (`skainet-backend-native-cpu`) installs the mapped-capable kernel packs and asserts
         * their derived encodings equal this set minus [TensorEncoding.Dense] (which is mapped
         * as an element view, not served by a matmul kernel) — update both together.
         */
        public val MAPPED_SERVABLE_DEFAULT: Set<TensorEncoding> = setOf(
            TensorEncoding.Dense(4),
            TensorEncoding.Q4_K, TensorEncoding.Q6_K, TensorEncoding.Q5_K,
            TensorEncoding.Q8_0, TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1,
        )

        /** The platform this code is running on. */
        public fun current(): StorageCapabilities = StorageCapabilities(
            supportsMappedFiles = PlatformStorage.supportsMappedFiles,
            supportsOffHeap = PlatformStorage.supports(MemoryDomain.HOST_OFFHEAP),
        )

        /** A JVM/native-class platform: everything works. */
        public val FULL: StorageCapabilities = StorageCapabilities(supportsMappedFiles = true)

        /** A browser-class platform: heap only, no mmap, no off-heap. */
        public val HEAP_ONLY: StorageCapabilities = StorageCapabilities(supportsMappedFiles = false, supportsOffHeap = false)
    }
}

/**
 * Decides where a tensor's bytes belong — the placement counterpart of [WeightFormResolver] (#1143,
 * closing the question #1133 asked).
 *
 * Same contract as the form resolver: a pure function of *what will be held* (the resolved
 * [WeightForm]), *what the profile says* ([PlannerProfile.domainFor], its thresholds) and *what the
 * platform can do* ([StorageCapabilities]). Consumers — the plan, the loader, a context — carry the
 * result; none of them decide. Nothing here allocates.
 */
public object AllocationResolver {

    /**
     * The allocation a weight needs once [PlanTensor.form] is honoured.
     *
     * Served from file-backed pages only when every condition holds: the form asks for
     * [WeightResidency.MAPPED], the platform can map, and the bytes really are the file's bytes —
     * a re-encoded ([EncodingRequest.DequantizeTo]/[EncodingRequest.RequantizeTo]) or re-ordered
     * ([WeightByteOrder.KERNEL_FEED]) weight is a load-time copy, and a copy cannot be paged from
     * the file it no longer matches. Everything else falls to [PlannerProfile.domainFor] over the
     * bytes actually held, so a dequantized giant goes off-heap and a small bias stays on it.
     */
    /**
     * Whether [weight] will really be served from file-backed pages: the form asks for
     * [WeightResidency.MAPPED], the platform can map, the bytes are the file's bytes (no
     * re-encode, no re-order), **and** the encoding is one the runtime can serve from a mapping
     * ([StorageCapabilities.mappedServableEncodings] — dense F32 since #921, Q4_K/Q6_K since
     * #1189). This is the predicate the plan uses to budget a weight against the page cache
     * instead of the heap, so it must not overclaim.
     */
    public fun servesFromMapping(
        weight: PlanTensor,
        platform: StorageCapabilities = StorageCapabilities.current(),
    ): Boolean {
        val form = weight.form
        val fileBytesAreTheBytes = form == null ||
            (form.encoding == EncodingRequest.KeepAsStored && form.order == WeightByteOrder.AS_STORED)
        return form?.residency == WeightResidency.MAPPED &&
            platform.supportsMappedFiles &&
            fileBytesAreTheBytes &&
            weight.format.encoding in platform.mappedServableEncodings
    }

    public fun resolve(
        weight: PlanTensor,
        profile: PlannerProfile,
        platform: StorageCapabilities = StorageCapabilities.current(),
    ): AllocationSpec {
        val mapped = servesFromMapping(weight, platform)
        val domain = if (mapped) MemoryDomain.MMAP_FILE else fallbackDomain(weight.residentBytes, profile, platform)
        return AllocationSpec(
            format = residentFormat(weight),
            elementCount = weight.elementCount,
            domain = domain,
            scope = ScopeKind.MODEL,
            mutable = false,
        )
    }

    /**
     * The allocation a transient (activation/scratch) tensor needs: never mapped, never
     * model-lifetime — only the profile's heap/off-heap threshold and the caller's scope.
     */
    public fun resolveTransient(
        format: Format,
        elementCount: Long,
        profile: PlannerProfile,
        platform: StorageCapabilities = StorageCapabilities.current(),
        scope: ScopeKind = ScopeKind.FORWARD,
    ): AllocationSpec {
        val bytes = format.physicalBytes(elementCount) ?: (format.dtype.sizeInBytes.toLong() * elementCount)
        return AllocationSpec(format, elementCount, fallbackDomain(bytes, profile, platform), scope, mutable = true)
    }

    /** The [Format] the weight holds once its form is honoured — dense after a dequantization. */
    public fun residentFormat(weight: PlanTensor): Format = when (val request = weight.form?.encoding) {
        null, EncodingRequest.KeepAsStored -> weight.format
        is EncodingRequest.DequantizeTo -> Format.dense(request.dtype)
        is EncodingRequest.RequantizeTo -> Format(weight.format.dtype, request.encoding)
    }

    /**
     * One line saying where [weight] lands and *why* — the transparency counterpart of
     * [resolve], for plan renders and load-time traces. The decision is recomputed, so the
     * explanation can never drift from what the resolver actually did.
     */
    public fun explain(
        weight: PlanTensor,
        profile: PlannerProfile,
        platform: StorageCapabilities = StorageCapabilities.current(),
    ): String {
        val spec = resolve(weight, profile, platform)
        val form = weight.form
        val why = when {
            spec.domain == MemoryDomain.MMAP_FILE ->
                "form asks MAPPED, platform can map, bytes are the file's bytes"
            form?.residency == WeightResidency.MAPPED && !platform.supportsMappedFiles ->
                "form asks MAPPED but this platform cannot map files"
            form?.residency == WeightResidency.MAPPED && form.encoding != EncodingRequest.KeepAsStored ->
                "form asks MAPPED but the weight is re-encoded at load — a copy cannot be paged from the file"
            form?.residency == WeightResidency.MAPPED && form.order == WeightByteOrder.KERNEL_FEED ->
                "form asks MAPPED but kernel-feed order is a load-time copy"
            form?.residency == WeightResidency.MAPPED &&
                weight.format.encoding !in platform.mappedServableEncodings ->
                "form asks MAPPED but no loader/kernel serves ${weight.format.encoding.name} " +
                    "from a mapping yet (see StorageCapabilities.mappedServableEncodings) — heap staging"
            else ->
                "resident ${MemoryPlans.formatBytes(weight.residentBytes)} vs off-heap threshold " +
                    MemoryPlans.formatBytes(profile.offHeapThresholdBytes) +
                    if (!platform.supportsOffHeap) " (no off-heap on this platform)" else ""
        }
        return "${weight.name}: ${spec.domain}/${spec.scope} — $why"
    }

    private fun fallbackDomain(bytes: Long, profile: PlannerProfile, platform: StorageCapabilities): MemoryDomain {
        val preferred = profile.domainFor(bytes)
        return if (preferred == MemoryDomain.HOST_OFFHEAP && !platform.supportsOffHeap) MemoryDomain.HOST_HEAP else preferred
    }
}
