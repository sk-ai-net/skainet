package sk.ainet.lang.memory.plan


/**
 * What a device has to offer, as **two pools** rather than one (SKEEP-002, #1038).
 *
 * On Android these are genuinely different resources: the ART managed heap is hard-capped per app
 * (256 MB by default, 512 MB with `largeHeap`) and is where every Kotlin array lives, while mapped
 * weights live in file-backed pages that do not count against that cap at all — they compete for
 * physical RAM, which the OS reclaims under pressure instead of killing the app. A planner that
 * totals one number cannot tell a model that will not fit from one that fits perfectly well as
 * long as its weights are mapped.
 *
 * @property totalRamBytes physical RAM (`ActivityManager.MemoryInfo.totalMem`)
 * @property availableRamBytes RAM the OS says is available now (`MemoryInfo.availMem`)
 * @property lowMemory the OS is already under pressure (`MemoryInfo.lowMemory`)
 * @property lowMemoryThresholdBytes below this the OS starts killing processes (`MemoryInfo.threshold`)
 * @property heapMaxBytes the managed-heap cap (`Runtime.maxMemory()`)
 * @property heapUsedBytes managed heap in use right now
 */
public data class DeviceMemory(
    val totalRamBytes: Long,
    val availableRamBytes: Long,
    val heapMaxBytes: Long,
    val heapUsedBytes: Long = 0L,
    val lowMemory: Boolean = false,
    val lowMemoryThresholdBytes: Long = 0L,
) {
    init {
        require(totalRamBytes >= 0 && availableRamBytes >= 0) { "RAM figures must be non-negative" }
        require(heapMaxBytes > 0) { "heapMaxBytes must be > 0" }
    }

    /** Managed heap still available to allocate into. */
    public val heapFreeBytes: Long get() = (heapMaxBytes - heapUsedBytes).coerceAtLeast(0L)

    /**
     * RAM that may be used without pushing the OS under its own low-memory threshold — the reserve
     * the device itself declares, floored at [RAM_RESERVE_FLOOR] for devices that report none.
     */
    public val usableRamBytes: Long
        get() = (availableRamBytes - maxOf(lowMemoryThresholdBytes, RAM_RESERVE_FLOOR)).coerceAtLeast(0L)

    public companion object {
        /** Reserve when the device reports no low-memory threshold. */
        public const val RAM_RESERVE_FLOOR: Long = 128L * MiB
    }
}

/** One resource pool of a [DeviceFit]: what the plan needs from it, and what it has. */
public data class PoolFit(val name: String, val neededBytes: Long, val budgetBytes: Long) {
    public val fits: Boolean get() = neededBytes <= budgetBytes
    /** Bytes left over (negative when it does not fit). */
    public val headroomBytes: Long get() = budgetBytes - neededBytes
}

/**
 * A [MemoryPlan] checked against a real device, pool by pool (M2-A5): the answer to "will this
 * model load on this phone", and when it will not, which pool ran out and what to do about it.
 *
 * @property weightsMapped whether the weights are loaded through `WeightResidency.MAPPED`, i.e. from
 *   file-backed pages that never count against the managed heap.
 */
public data class DeviceFit(
    val plan: MemoryPlan,
    val device: DeviceMemory,
    val weightsMapped: Boolean,
    val heap: PoolFit,
    val ram: PoolFit,
    val suggestions: List<Suggestion>,
) {
    /** True only when *both* pools have room. */
    public val fits: Boolean get() = heap.fits && ram.fits

    /** The pool that ran out, or `null` when the plan fits. */
    public val blockingPool: String?
        get() = when {
            !heap.fits -> heap.name
            !ram.fits -> ram.name
            else -> null
        }

    public fun render(): String = buildString {
        append(plan.input.modelName); append(" · ctx "); append(plan.input.ctx)
        append(if (weightsMapped) " · weights mapped\n" else " · weights on the heap\n")
        for (p in listOf(heap, ram)) {
            append("  "); append(p.name.padEnd(14))
            append(MemoryPlans.formatBytes(p.neededBytes).padStart(10))
            append(" of "); append(MemoryPlans.formatBytes(p.budgetBytes).padStart(10))
            append(if (p.fits) "   ✔" else "   ✘ short by ${MemoryPlans.formatBytes(-p.headroomBytes)}")
            append('\n')
        }
        if (device.lowMemory) append("  device reports low memory — the OS is already reclaiming\n")
        if (!fits && suggestions.isNotEmpty()) {
            append("  suggestions: ")
            append(suggestions.joinToString(" · ") { "${it.text} (−${MemoryPlans.formatBytes(it.savesBytes)})" })
            append('\n')
        }
    }
}

/**
 * Check [plan] against [device]'s two pools (SKEEP-002, M2-A5).
 *
 * The managed heap carries the KV cache, the forward slab and the plan's heap headroom — plus the
 * weights when they are *not* mapped, which is exactly what makes a 600 MB Q4 model impossible
 * under a 512 MB cap and unremarkable when mapped. Physical RAM carries everything, mapped pages
 * included: they are evictable, not free.
 *
 * @param weightsMapped weights come from file-backed pages (`WeightResidency.MAPPED`)
 */
public fun MemoryPlan.fitOn(device: DeviceMemory, weightsMapped: Boolean): DeviceFit {
    val heapNeeded = kvBytes + forwardBytes + headroomBytes + if (weightsMapped) 0L else weightsBytes
    val heap = PoolFit("managed heap", heapNeeded, device.heapFreeBytes)
    val ram = PoolFit("device RAM", totalBytes, device.usableRamBytes)

    val suggestions = ArrayList<Suggestion>()
    if (!heap.fits && !weightsMapped) {
        suggestions += Suggestion(
            "load with staging = MAPPED (weights move to file-backed pages, off the managed heap)",
            weightsBytes,
        )
    }
    // The plan's own advice (smaller ctx, quantized KV, a smaller model) is priced against a
    // budget, and the budget here is "whatever the tighter pool is short by": a plan that misses
    // the heap cap by 200 MB has to save 200 MB, wherever those bytes were going to sit.
    val deficit = maxOf(-heap.headroomBytes, -ram.headroomBytes)
    if (deficit > 0) {
        val target = (totalBytes - deficit).coerceAtLeast(0L)
        val pool = if (!heap.fits) heap.name else ram.name
        suggestions += copy(budget = Budget(target, "$pool on this device")).suggestions()
    }
    return DeviceFit(this, device, weightsMapped, heap, ram, suggestions)
}
