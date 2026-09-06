package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.AllocationSpec
import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * One weight tensor as the planner sees it — shape and format only, never bytes (SKEEP-003 §8 item 1,
 * PRD M0-F1). [id] is the [TensorId] the checkpoint's `NameMap` produced, `null` when unmapped.
 */
public data class PlanTensor(
    val name: String,
    val id: TensorId?,
    val format: Format,
    val elementCount: Long,
    /** Physical bytes; falls back to the checkpoint's own byte count when the encoding cannot compute it. */
    val bytes: Long,
    /**
     * The form this weight was resolved to take in memory (#1109/#1116), or `null` when nothing has
     * resolved one and the file's own bytes are what will be held.
     */
    val form: WeightForm? = null,
) {
    /**
     * What this weight actually costs in memory once [form] is honoured.
     *
     * [bytes] is what the *file* holds. They differ exactly when the resolved form re-encodes:
     * a Q4_K tensor kept as Q4_K costs its packed bytes, and the same tensor dequantized to FP32
     * costs roughly eight times that. Planning the first number while loading the second is how a
     * dequantization becomes an OOM instead of a line in a table.
     */
    val residentBytes: Long
        get() = when (val request = form?.encoding) {
            null, EncodingRequest.KeepAsStored -> bytes
            is EncodingRequest.DequantizeTo ->
                Format.dense(request.dtype).physicalBytes(elementCount)
                    ?: (request.dtype.sizeInBytes.toLong() * elementCount)
            is EncodingRequest.RequantizeTo ->
                Format(format.dtype, request.encoding).physicalBytes(elementCount) ?: bytes
        }

    /**
     * The allocation this weight needs — resolved, not assumed (#1143). The old form of this
     * property hardcoded mapped/model-lifetime for every weight regardless of what [form] asked,
     * what the profile said, or whether the platform could map at all.
     */
    public fun allocation(
        profile: PlannerProfile,
        platform: StorageCapabilities = StorageCapabilities.current(),
    ): AllocationSpec = AllocationResolver.resolve(this, profile, platform)
}

/** The transformer geometry the KV-cache and forward-slab estimates need (from the GGUF header). */
public data class ModelGeometry(
    val layers: Int,
    val heads: Int,
    val kvHeads: Int,
    /** Per-head key width (`attention.key_length`, = embedding / heads when absent). */
    val headDim: Int,
    /** Per-head value width (`attention.value_length`, = [headDim] when absent). */
    val valueDim: Int = headDim,
    val embeddingLength: Int,
    /** Feed-forward width (`feed_forward_length`); 4 × embedding when absent. */
    val feedForwardLength: Int = 4 * embeddingLength,
    val vocabSize: Int,
    /** The context length the model was trained for, if the header says. */
    val trainedContextLength: Int? = null,
)

/** How the KV cache is stored. */
public enum class KvCacheMode(public val label: String) {
    /**
     * 4 bytes per element — what `DefaultKvCacheStore` actually stores today (dense FloatArray
     * rings). Planning a dense store as [BF16] understates it by 2×, which is the kind of drift
     * the plan-vs-actual check exists to catch (#1074).
     */
    FP32("fp32"),

    /** 2 bytes per element (bf16/f16) — a narrow-float KV store. */
    BF16("bf16"),
    /** TurboQuant 4-bit polar codes, block 128 (decision #11 default when the plan is tight). */
    TURBOQUANT_4("TurboQuant 4-bit");

    /** Bytes for [elements] cache elements under this mode. */
    public fun bytes(elements: Long): Long = when (this) {
        FP32 -> 4L * elements
        BF16 -> 2L * elements
        TURBOQUANT_4 -> TensorEncoding.TurboQuantPolar(bitsPerElement = 4, blockSize = 128).physicalBytes(elements) ?: (elements / 2)
    }
}

/** What to plan for: the model (header only), the context length and the prefill chunk. */
/**
 * The KV byte width taken from a store's declared `Format` — what the planner should use instead of
 * guessing a [KvCacheMode] (#1077). A dense FP32 ring reports 4 bytes per element, a bf16 ring 2, a
 * TurboQuant ring its packed width.
 */
public fun kvBytesFor(format: Format, elements: Long): Long =
    format.physicalBytes(elements) ?: (format.dtype.sizeInBytes.toLong() * elements)

public data class PlanInput(
    val modelName: String,
    val architecture: String,
    val weights: List<PlanTensor>,
    val geometry: ModelGeometry?,
    val ctx: Int,
    val prefillChunk: Int = DEFAULT_PREFILL_CHUNK,
    val kvMode: KvCacheMode = KvCacheMode.BF16,
) {
    init { require(ctx > 0) { "ctx must be > 0" }; require(prefillChunk > 0) { "prefillChunk must be > 0" } }

    /** Names the checkpoint's name map could not translate — never dropped (M0-F4). */
    val unmappedWeights: List<String> get() = weights.filter { it.id == null }.map { it.name }

    public companion object {
        public const val DEFAULT_PREFILL_CHUNK: Int = 256
    }
}

/**
 * Memory budget the plan is checked against (decision #11): an explicit number of bytes, or
 * `available − reserve` for a platform.
 */
public data class Budget(val bytes: Long, val description: String) {
    public companion object {
        /** Reserve the OS/app needs on Android and desktop JVMs (decision #11). */
        public const val RESERVE_ANDROID_JVM: Long = 700L * MiB
        /** Reserve on Kotlin/Native targets (decision #11). */
        public const val RESERVE_NATIVE: Long = 300L * MiB

        public fun of(bytes: Long): Budget = Budget(bytes, "explicit")
        public fun available(availableBytes: Long, reserve: Long = RESERVE_ANDROID_JVM): Budget =
            Budget((availableBytes - reserve).coerceAtLeast(0), "available − reserve (${reserve / MiB} MB)")
    }
}

/**
 * One line of the plan: what, how much, and whether it is resident for the whole session.
 * A [mapped] line is file-backed page cache — resident in RSS but evictable, and **not** charged
 * against the heap [Budget] (#1189: a mapped 1 GB model runs under a 256 MB ART cap).
 */
public data class PlanLine(
    val section: String,
    val detail: String,
    val bytes: Long,
    val resident: Boolean,
    val mapped: Boolean = false,
)

/** A concrete way to make a plan fit, with the bytes it saves. */
public data class Suggestion(val text: String, val savesBytes: Long)

/**
 * The memory plan of a model at a context length: weights (resident), KV cache, forward slab and
 * heap headroom, totalled against a [Budget]. Pure arithmetic over [PlanInput]; nothing is
 * allocated (PRD M0-F1..F3). The estimates are deliberately simple and documented in
 * [MemoryPlans]; milestone M1's plan-vs-actual check calibrates them against real allocations.
 */
public data class MemoryPlan(
    val input: PlanInput,
    val weightsBytes: Long,
    val kvBytes: Long,
    /** KV bytes under the *other* mode, so the table can show both (bf16 vs TurboQuant). */
    val kvBytesAlternate: Long,
    val forwardBytes: Long,
    val headroomBytes: Long,
    val budget: Budget?,
    /**
     * What the weights occupy *in the file*, before any resolved [WeightForm] re-encodes them
     * (#1116). Equal to [weightsBytes] when nothing was resolved, which is every plan built before
     * forms existed — hence the default.
     */
    val weightsAsStoredBytes: Long = weightsBytes,
    /**
     * The part of [weightsBytes] served from file-backed pages (#1189): resident in RSS but
     * evictable and outside the managed heap, so it is charged against device RAM/page cache
     * rather than the [budget]. Zero for a fully heap-staged model — every plan before #1189.
     */
    val weightsMappedBytes: Long = 0L,
) {
    /**
     * Bytes the resolved forms add to the weights — a dequantization's price, made visible before
     * it is paid rather than discovered as an OOM. Zero when the weights are held as stored.
     */
    val formConversionBytes: Long get() = weightsBytes - weightsAsStoredBytes

    /** The part of [weightsBytes] that really lands on the managed heap. */
    val weightsHeapBytes: Long get() = weightsBytes - weightsMappedBytes

    /** The full footprint — heap sections plus mapped pages. What RSS converges to, not what the heap holds. */
    val totalBytes: Long get() = weightsBytes + kvBytes + forwardBytes + headroomBytes

    /**
     * What is charged against the [budget]: everything except the mapped weight pages, which the
     * OS pages in and evicts against device RAM (#1189 measured: 1.0 GB mapped, 566 KB heap,
     * decode under a 256 MB cap). Equal to [totalBytes] when nothing is mapped.
     */
    val budgetedBytes: Long get() = totalBytes - weightsMappedBytes
    val residentBytes: Long get() = weightsBytes + kvBytes

    /** `true` when a budget is set and the budget-charged total fits; `null` without a budget. */
    val fits: Boolean? get() = budget?.let { budgetedBytes <= it.bytes }

    val lines: List<PlanLine>
        get() = buildList {
            if (weightsMappedBytes > 0L) {
                add(PlanLine("weights", "mapped, as stored", weightsMappedBytes, resident = true, mapped = true))
                if (weightsHeapBytes > 0L) {
                    add(
                        PlanLine(
                            "weights",
                            if (formConversionBytes == 0L) "heap-staged, packed"
                            else "re-encoded at load (+${MemoryPlans.formatBytes(formConversionBytes)})",
                            weightsHeapBytes,
                            resident = true,
                        ),
                    )
                }
            } else {
                add(
                    PlanLine(
                        "weights",
                        if (formConversionBytes == 0L) "heap-staged, packed"
                        else "re-encoded at load (+${MemoryPlans.formatBytes(formConversionBytes)})",
                        weightsBytes,
                        resident = true,
                    ),
                )
            }
            add(PlanLine("kv cache", input.kvMode.label + " @ ctx ${input.ctx}", kvBytes, resident = true))
            add(PlanLine("forward", "prefill chunk ${input.prefillChunk}", forwardBytes, resident = false))
            add(PlanLine("heap", "headroom", headroomBytes, resident = false))
        }

    /** At least two concrete suggestions with their savings when the plan does not fit (M0-F3). */
    public fun suggestions(): List<Suggestion> {
        val b = budget ?: return emptyList()
        if (budgetedBytes <= b.bytes) return emptyList()
        val out = ArrayList<Suggestion>()
        if (input.kvMode == KvCacheMode.BF16 && kvBytesAlternate < kvBytes) {
            out += Suggestion("--kv turboquant", kvBytes - kvBytesAlternate)
        }
        val halfCtx = (input.ctx / 2).coerceAtLeast(1)
        if (halfCtx < input.ctx) {
            val half = MemoryPlans.plan(input.copy(ctx = halfCtx), budget)
            out += Suggestion("--ctx $halfCtx", budgetedBytes - half.budgetedBytes)
        }
        if (formConversionBytes > 0) {
            // Worth saying first: unlike ctx or KV mode, this cost was not asked for — it is what
            // the resolver chose because no kernel on the target could feed the stored encoding.
            out += Suggestion(
                "a build with kernels for the stored encoding: loading it as-is instead of re-encoding",
                formConversionBytes,
            )
        }
        val over = budgetedBytes - b.bytes
        out += Suggestion("a smaller model: weights must shrink by ≥ ${MemoryPlans.formatBytes(over)} (e.g. a lower-bit quantization of the same model)", over)
        return out
    }

    /** The PRD §4.3 table. */
    public fun render(): String = buildString {
        val g = input.geometry
        append(input.modelName); append(" · "); append(input.architecture)
        if (g != null) { append(" · "); append(g.layers); append(" layers") }
        append(" · ctx "); append(input.ctx); append('\n')
        for (l in lines) {
            append("  "); append(l.section.padEnd(10)); append(l.detail.padEnd(26)); append(MemoryPlans.formatBytes(l.bytes).padStart(10))
            if (l.mapped) append("   mapped (page cache, evictable — not heap)")
            else if (l.resident) append("   resident")
            if (l.section == "kv cache") append("   (").append(MemoryPlans.formatBytes(kvBytesAlternate)).append(" with ").append(if (input.kvMode == KvCacheMode.TURBOQUANT_4) KvCacheMode.BF16.label else KvCacheMode.TURBOQUANT_4.label).append(')')
            append('\n')
        }
        val budgetLabel = if (weightsMappedBytes > 0L) "total heap" else "total"
        append("  "); append(budgetLabel.padEnd(36)); append(MemoryPlans.formatBytes(budgetedBytes).padStart(10))
        val b = budget
        if (b != null) {
            append("   of "); append(MemoryPlans.formatBytes(b.bytes)); append(if (fits == true) "  ✔ fits" else "  ✘ does not fit")
            append('\n')
            if (weightsMappedBytes > 0L) {
                append("  mapped weights (").append(MemoryPlans.formatBytes(weightsMappedBytes))
                append(") page against device RAM, not this budget (#1189)\n")
            }
            val s = suggestions()
            if (s.isNotEmpty()) {
                append("  suggestions: "); append(s.joinToString(" · ") { "${it.text} (−${MemoryPlans.formatBytes(it.savesBytes)})" }); append('\n')
            }
        } else append('\n')
        val unmapped = input.unmappedWeights
        if (unmapped.isNotEmpty()) {
            append("  unmapped tensors (kept, not identified): "); append(unmapped.size); append(" — "); append(unmapped.take(5).joinToString(", ")); if (unmapped.size > 5) append(", …"); append('\n')
        }
    }
}

/** The arithmetic behind [MemoryPlan]. */
public object MemoryPlans {

    /** Fixed heap headroom the JVM/ART runtime needs besides tensors (decision #11 profile). */
    public const val HEAP_HEADROOM_BYTES: Long = 64L * MiB

    /**
     * Build the plan. Estimates:
     * - weights: sum of the packed byte sizes (they are touched every token, so counted resident) —
     *   split by [AllocationResolver.servesFromMapping] into heap-charged bytes and mapped bytes,
     *   because a mapped weight pages against device RAM, not the budget (#1189);
     * - kv cache: `layers × 2 × ctx × kvHeads × (headDim + valueDim)/2 × mode bytes`;
     * - forward slab for a chunk of `T = min(prefillChunk, ctx)` tokens, FP32:
     *   `T × (4·emb + 3·ffn + heads·ctx) × 4 B` (residual stream, attention projections, gated FFN
     *   intermediates, attention scores over the context) plus one `vocab × 4 B` logits row;
     * - heap headroom: [HEAP_HEADROOM_BYTES].
     */
    public fun plan(
        input: PlanInput,
        budget: Budget? = null,
        platform: StorageCapabilities = StorageCapabilities.current(),
    ): MemoryPlan {
        val weights = input.weights.sumOf { it.residentBytes }
        val weightsMapped = input.weights
            .filter { AllocationResolver.servesFromMapping(it, platform) }
            .sumOf { it.residentBytes }
        val weightsAsStored = input.weights.sumOf { it.bytes }
        val g = input.geometry
        val kvElements = if (g != null) kvElements(g, input.ctx) else 0L
        val kv = input.kvMode.bytes(kvElements)
        val kvAlt = (if (input.kvMode == KvCacheMode.TURBOQUANT_4) KvCacheMode.BF16 else KvCacheMode.TURBOQUANT_4).bytes(kvElements)
        val forward = if (g != null) forwardBytes(g, input.ctx, input.prefillChunk) else 0L
        return MemoryPlan(input, weights, kv, kvAlt, forward, HEAP_HEADROOM_BYTES, budget, weightsAsStored, weightsMapped)
    }

    public fun kvElements(g: ModelGeometry, ctx: Int): Long =
        g.layers.toLong() * ctx * g.kvHeads * (g.headDim + g.valueDim)

    public fun forwardBytes(g: ModelGeometry, ctx: Int, prefillChunk: Int): Long {
        val t = minOf(prefillChunk, ctx).toLong()
        val perToken = 4L * g.embeddingLength + 3L * g.feedForwardLength + g.heads.toLong() * ctx
        return t * perToken * 4L + g.vocabSize.toLong() * 4L
    }

    public fun formatBytes(bytes: Long): String = when {
        bytes >= GiB -> formatOneDecimal(bytes.toDouble() / GiB) + " GB"
        bytes >= MiB -> (bytes / MiB).toString() + " MB"
        bytes >= KiB -> (bytes / KiB).toString() + " KB"
        else -> "$bytes B"
    }

    private fun formatOneDecimal(v: Double): String {
        val tenths = kotlin.math.round(v * 10).toLong()
        return "${tenths / 10}.${tenths % 10}"
    }
}

internal const val KiB: Long = 1024L
internal const val MiB: Long = 1024L * 1024L
internal const val GiB: Long = 1024L * 1024L * 1024L
