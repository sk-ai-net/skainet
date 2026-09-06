package sk.ainet.io.gguf

import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.plan.AllocationResolver
import sk.ainet.lang.memory.plan.KernelCapabilities
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.memory.plan.ProfiledPlan
import sk.ainet.lang.memory.plan.StorageCapabilities
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.resolveWeightForms
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceSink

/**
 * Plan → load, wired (#1144): the loader consults what the resolvers decided, instead of the two
 * pipelines sharing vocabulary and never talking.
 *
 * [resolve] reads the GGUF *header* (no payload), resolves every weight's [WeightForm] from
 * file × [PlannerProfile] × [KernelCapabilities], applies the caller's per-tensor overrides,
 * and returns a [Resolution]: a loader that will deliver exactly those forms, plus the resolved
 * plan input so the same decisions are priceable ([Resolution.profiledPlan]) and explainable
 * ([Resolution.explainPlacements]) *before a byte of payload is read*.
 *
 * ## Who decides — the precedence order
 *
 * **Your override > the resolver.** [overrideFormFor] outranks everything for the tensors it names,
 * including the deliberately blunt `WeightForm(DequantizeTo(FP32), residency = HEAP)` — everything
 * dense, on the managed heap. Tensors it returns `null` for get the resolver's answer. The plan and
 * the explanations are computed *after* overrides are applied, so what you print is what you load.
 */
public object ResolvedGguf {

    /** What [resolve] decided: the loader that obeys it, and the resolved input that explains it. */
    public data class Resolution(
        val loader: StreamingGgufParametersLoader,
        /** The header-derived plan input with every weight's resolved (and overridden) form. */
        val input: PlanInput,
        val profile: PlannerProfile,
        val platform: StorageCapabilities,
    ) {
        /** The plan these forms cost against [availableBytes] — `requireFits()` refuses pre-load (M2-F6). */
        public fun profiledPlan(availableBytes: Long): ProfiledPlan = profile.plan(input, availableBytes)

        /** One line per weight: where it lands and why — [AllocationResolver.explain] over the resolved forms. */
        public fun explainPlacements(): List<String> =
            input.weights.map { AllocationResolver.explain(it, profile, platform) }
    }

    /**
     * Resolve every weight's form from the header, apply [overrideFormFor], and build the loader
     * that delivers it. Reads header only; costs a few kilobytes.
     *
     * @param capabilities what the backend's kernels can feed — pass the registry-backed
     *   implementation from the backend in use, or [KernelCapabilities.DENSE_ONLY] to price the
     *   worst case
     * @param overrideFormFor the user-wins channel; `null` per tensor means "resolver decides"
     * @throws IllegalStateException when the profile is strict and a weight would dequantize
     */
    public fun resolve(
        sourceProvider: () -> RandomAccessSource,
        profile: PlannerProfile,
        capabilities: KernelCapabilities,
        ctx: Int? = null,
        platform: StorageCapabilities = StorageCapabilities.current(),
        overrideFormFor: (tensorName: String) -> WeightForm? = { null },
        onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        traceSink: TraceSink = NoopTraceSink,
    ): Resolution {
        val resolved = sourceProvider().use { source ->
            StreamingGGUFReader.open(source).planInput(ctx)
        }.resolveWeightForms(profile, capabilities)
        val overridden = resolved.copy(
            weights = resolved.weights.map { w -> overrideFormFor(w.name)?.let { w.copy(form = it) } ?: w },
        )
        val forms: Map<String, WeightForm?> = overridden.weights.associate { it.name to it.form }
        val loader = StreamingGgufParametersLoader(
            sourceProvider = sourceProvider,
            onProgress = onProgress,
            weightFormFor = { name -> forms[name] },
            traceSink = traceSink,
        )
        return Resolution(loader, overridden, profile, platform)
    }
}
