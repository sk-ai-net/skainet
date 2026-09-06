package sk.ainet.lang.memory.plan

import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * The rules a planner applies for a class of device (SKEEP-003 §8 item 1, decision #11; M2-F6).
 *
 * A 2 GB phone and a workstation do not want the same defaults, and the difference is not a matter
 * of taste: on the phone the reserve is large relative to the budget, weights *must* be mapped, the
 * KV cache has to shrink before the model does, and a dispatcher-inserted dequantization is a bug
 * rather than a slow path. A profile makes those rules explicit and testable instead of leaving
 * them as folklore in whoever configures the loader.
 *
 * @property reserveBytes what the OS and the rest of the app need; the budget is `available − this`
 * @property prefillChunk tokens per prefill step — the width the forward slab is pre-sized for
 * @property kvMode the KV format a comfortable plan uses
 * @property kvAutoQuantizeAbove switch KV to [KvCacheMode.TURBOQUANT_4] when the plan needs more
 *   than this fraction of the budget (`1.0` disables the rule)
 * @property offHeapThresholdBytes allocations of at least this many bytes belong off the heap
 * @property dequantWarnFraction dispatcher-inserted dequantization above this share of the bytes a
 *   decode step reads is worth a warning — it means a kernel is missing for the format on disk
 * @property strict turn those warnings into failures
 * @property weightsMapped weights are expected to load through `WeightResidency.MAPPED`
 */
public data class PlannerProfile(
    val name: String,
    val reserveBytes: Long,
    val prefillChunk: Int = PlanInput.DEFAULT_PREFILL_CHUNK,
    val kvMode: KvCacheMode = KvCacheMode.BF16,
    val kvAutoQuantizeAbove: Double = 1.0,
    val offHeapThresholdBytes: Long = OFF_HEAP_THRESHOLD,
    val dequantWarnFraction: Double = 0.05,
    val strict: Boolean = false,
    val weightsMapped: Boolean = false,
) {
    init {
        require(reserveBytes >= 0) { "reserveBytes must be >= 0" }
        require(prefillChunk > 0) { "prefillChunk must be > 0" }
        require(kvAutoQuantizeAbove > 0.0) { "kvAutoQuantizeAbove must be > 0" }
        require(offHeapThresholdBytes > 0) { "offHeapThresholdBytes must be > 0" }
    }

    /** The budget this profile derives from [availableBytes]. */
    public fun budget(availableBytes: Long): Budget =
        Budget((availableBytes - reserveBytes).coerceAtLeast(0L), "$name: available − ${MemoryPlans.formatBytes(reserveBytes)}")

    /** Where an allocation of [bytes] belongs — the heap/off-heap threshold (decision #11). */
    public fun domainFor(bytes: Long): MemoryDomain =
        if (bytes >= offHeapThresholdBytes) MemoryDomain.HOST_OFFHEAP else MemoryDomain.HOST_HEAP

    /**
     * Plan [input] under this profile against [availableBytes].
     *
     * The profile's prefill chunk and KV mode replace whatever the input carried, and if the
     * resulting plan needs more than [kvAutoQuantizeAbove] of the budget, the KV cache is
     * re-planned as TurboQuant-4 *before* anything else is suggested — quantizing the cache is the
     * cheapest thing to give up, and doing it automatically is the difference between a model that
     * loads and a error message on a phone.
     */
    public fun plan(input: PlanInput, availableBytes: Long): ProfiledPlan {
        val budget = budget(availableBytes)
        val base = MemoryPlans.plan(input.copy(prefillChunk = prefillChunk, kvMode = kvMode), budget)
        val notes = ArrayList<String>()
        var plan = base
        // Budget pressure comes from what is charged against the budget — mapped weights page
        // against device RAM, not the heap (#1189), so they must not trigger KV quantization.
        val share = if (budget.bytes > 0) base.budgetedBytes.toDouble() / budget.bytes else Double.MAX_VALUE
        if (kvMode != KvCacheMode.TURBOQUANT_4 && share > kvAutoQuantizeAbove) {
            val quantized = MemoryPlans.plan(base.input.copy(kvMode = KvCacheMode.TURBOQUANT_4), budget)
            if (quantized.budgetedBytes < base.budgetedBytes) {
                plan = quantized
                notes += "KV cache switched to ${KvCacheMode.TURBOQUANT_4.label}: the plan needed " +
                    "${percent(share)} of the budget (over ${percent(kvAutoQuantizeAbove)}), saving " +
                    MemoryPlans.formatBytes(base.kvBytes - quantized.kvBytes)
            }
        }
        if (weightsMapped) notes += "weights are counted resident and mapped — off the managed heap"
        return ProfiledPlan(this, plan, notes)
    }

    /**
     * The verdict on how much the dispatcher had to dequantize during decode, given the share of
     * bytes read that went through an adapter (`GenerationMetrics.adapterShareOfBytesRead`).
     *
     * A packed weight that has to be widened before every matmul means the kernel for its format is
     * missing — the cost is real memory traffic, not a rounding error, which is why this profile
     * can be run [strict] in CI and lenient on a desktop.
     */
    public fun checkDequant(adapterShareOfBytesRead: Double?): DequantVerdict {
        val share = adapterShareOfBytesRead ?: return DequantVerdict(this, 0.0, DequantSeverity.OK, "no bytes read")
        if (share <= dequantWarnFraction) {
            return DequantVerdict(this, share, DequantSeverity.OK, "adapters moved ${percent(share)} of the bytes read")
        }
        val message = "adapters moved ${percent(share)} of the bytes read, over the ${percent(dequantWarnFraction)} " +
            "$name limit — a kernel for the on-disk format is missing and every step pays for it"
        return DequantVerdict(this, share, if (strict) DequantSeverity.ERROR else DequantSeverity.WARN, message)
    }

    /** This profile with [strict] on — what a CI run or an acceptance test uses. */
    public fun strict(): PlannerProfile = copy(name = "$name (strict)", strict = true)

    public companion object {
        /** Off-heap threshold from decision #11: below this, a heap array is cheaper than a mapping. */
        public const val OFF_HEAP_THRESHOLD: Long = 256L * 1024

        /**
         * A 2 GB-class phone: 700 MB reserved for the OS and the app, weights mapped, KV
         * automatically quantized once the plan passes 80 % of the budget, and dequantization
         * treated as the defect it is — [strict], so a missing kernel fails rather than quietly
         * costing several times the weight's size. The default on Android.
         *
         * Use `copy(strict = false)` for a build that would rather load slowly than not at all.
         */
        public val MOBILE_2GB: PlannerProfile = PlannerProfile(
            name = "mobile-2gb",
            reserveBytes = Budget.RESERVE_ANDROID_JVM,
            prefillChunk = PlanInput.DEFAULT_PREFILL_CHUNK,
            kvMode = KvCacheMode.BF16,
            kvAutoQuantizeAbove = 0.80,
            weightsMapped = true,
            // This profile has always *said* dequantization is "the defect it is"; the flag said
            // otherwise. On a 2 GB board a missing kernel is not a slow path to take quietly — it
            // is a weight arriving several times its size on the device least able to hold it, and
            // the honest moment to say so is before the load rather than at the OOM.
            strict = true,
        )

        /**
         * An embedded device with limited memory and compute (#1169): the number the caller passes
         * as available bytes IS the usable RAM — already net of what the OS holds — so
         * [reserveBytes] is deliberately **zero and must stay zero**; adding a reserve here would
         * double-count what the caller already subtracted. Weights are counted mapped, the KV
         * cache auto-quantizes past 80 % of the budget, and the profile is not strict: this
         * profile's job is a pre-flight verdict, not a refusal.
         */
        public val EDGE: PlannerProfile = PlannerProfile(
            name = "edge",
            reserveBytes = 0,
            kvAutoQuantizeAbove = 0.80,
            weightsMapped = true,
        )

        /** A desktop or server JVM: the same reserve, no automatic KV quantization, heap staging. */
        public val DESKTOP: PlannerProfile = PlannerProfile(
            name = "desktop",
            reserveBytes = Budget.RESERVE_ANDROID_JVM,
        )

        /** Kotlin/Native hosts, which reserve less than a JVM (decision #11). */
        public val NATIVE: PlannerProfile = PlannerProfile(
            name = "native",
            reserveBytes = Budget.RESERVE_NATIVE,
        )

        /** The profile a device's own numbers call for: mobile below [MOBILE_RAM_CEILING] of RAM. */
        public fun forDevice(device: DeviceMemory): PlannerProfile =
            if (device.totalRamBytes <= MOBILE_RAM_CEILING) MOBILE_2GB else DESKTOP

        /** Devices with at most this much RAM get [MOBILE_2GB]. */
        public const val MOBILE_RAM_CEILING: Long = 3L * 1024 * MiB
    }
}

private fun percent(fraction: Double): String {
    val scaled = (fraction * 1000).toLong()
    return "${scaled / 10}.${scaled % 10}%"
}

/** How bad a dequantization share is under a profile. */
public enum class DequantSeverity { OK, WARN, ERROR }

/** The verdict of [PlannerProfile.checkDequant]. */
public data class DequantVerdict(
    val profile: PlannerProfile,
    val share: Double,
    val severity: DequantSeverity,
    val message: String,
) {
    /** Throw when this profile is strict and the share is over its limit. */
    public fun requireAcceptable() {
        if (severity == DequantSeverity.ERROR) throw IllegalStateException(message)
    }
}

/** A [MemoryPlan] made under a [PlannerProfile], with what the profile decided along the way. */
public data class ProfiledPlan(
    val profile: PlannerProfile,
    val plan: MemoryPlan,
    val notes: List<String>,
) {
    val fits: Boolean? get() = plan.fits

    /**
     * Refuse before anything is allocated (M2-F6): throws when the plan does not fit its budget, or
     * when [device] is given and either of its pools is short.
     */
    public fun requireFits(device: DeviceMemory? = null) {
        if (device != null) {
            val fit = plan.fitOn(device, profile.weightsMapped)
            if (!fit.fits) throw IllegalStateException("does not fit this device under profile '${profile.name}':\n" + fit.render())
            return
        }
        if (plan.fits == false) throw IllegalStateException("does not fit the budget under profile '${profile.name}':\n" + plan.render())
    }

    public fun render(): String = buildString {
        append("profile ").append(profile.name)
        append(" · reserve ").append(MemoryPlans.formatBytes(profile.reserveBytes))
        append(" · prefill ").append(profile.prefillChunk)
        append(" · off-heap ≥ ").append(MemoryPlans.formatBytes(profile.offHeapThresholdBytes))
        if (profile.kvAutoQuantizeAbove < 1.0) append(" · KV auto-quantize over ").append(percent(profile.kvAutoQuantizeAbove))
        if (profile.strict) append(" · strict")
        append('\n')
        for (n in notes) append("  note: ").append(n).append('\n')
        append(plan.render())
    }
}
