
package sk.ainet.apps.plan

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.planInput
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.KvCacheMode
import sk.ainet.lang.memory.plan.MemoryPlan
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.KernelCapabilities
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.memory.plan.resolveWeightForms
import sk.ainet.lang.memory.plan.ProfiledPlan
import java.io.File
import kotlin.system.exitProcess

/**
 * `skainet plan <model.gguf> [--ctx N] [--budget 1.3G] [--kv bf16|turboquant] [--profile P] [--list <glob>]`
 *
 * Milestone M0 of SKEEP-003 ("know before you load"): prints the memory plan of a GGUF model at a
 * context length — weights (resident), KV cache, forward slab, heap headroom — against a budget,
 * with concrete suggestions when it does not fit, and lists tensors by `TensorId`. Reads the GGUF
 * header only; no tensor bytes are touched.
 *
 * `--profile` (M2-F6, #1039) plans under a device profile instead of raw defaults: `mobile` is the
 * 2 GB phone (700 MB reserved, weights mapped, KV quantized automatically once the plan passes 80 %
 * of the budget), `desktop` keeps today's behaviour, `native` reserves the smaller Kotlin/Native
 * amount. The profile and every decision it made are printed above the table, so a plan can be read
 * back later without guessing which rules produced it.
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-plan")
    val model by parser.argument(ArgType.String, fullName = "model", description = "Path to the GGUF file")
    val ctx by parser.option(ArgType.Int, fullName = "ctx", description = "Context length to plan for (default: the model's trained context length, or 2048)")
    val budget by parser.option(ArgType.String, fullName = "budget", description = "Memory budget, e.g. 1.3G, 900M, 1500000000; default: JVM max heap + direct memory estimate")
    val kv by parser.option(ArgType.Choice(listOf("bf16", "turboquant"), { it }), fullName = "kv", description = "KV cache mode").default("bf16")
    val prefill by parser.option(ArgType.Int, fullName = "prefill-chunk", description = "Prefill chunk size for the forward slab").default(PlanInput.DEFAULT_PREFILL_CHUNK)
    val list by parser.option(ArgType.String, fullName = "list", description = "List tensors whose TensorId matches this glob, e.g. 'model.layers[3].*'")
    val noBudget by parser.option(ArgType.Boolean, fullName = "no-budget", description = "Print the plan without a fit check").default(false)
    val profileName by parser.option(
        ArgType.Choice(listOf("none", "mobile", "desktop", "native"), { it }),
        fullName = "profile",
        description = "Device profile whose rules the plan follows (M2-F6): mobile = 2 GB phone, desktop, native",
    ).default("none")
    val kernels by parser.option(
        ArgType.Choice(listOf("all", "dense"), { it }),
        fullName = "kernels",
        description = "What the target's kernels can feed (#1116): all = packed kernels for every " +
            "shipped encoding (the default, and what SKaiNET's CPU backend carries); dense = FP32 " +
            "only, so the plan prices dequantizing every quantized weight at load",
    ).default("all")
    parser.parse(args)

    val file = File(model)
    if (!file.isFile) { System.err.println("skainet plan: file not found: $model"); exitProcess(2) }

    val kvMode = if (kv == "turboquant") KvCacheMode.TURBOQUANT_4 else KvCacheMode.BF16
    val profile = profileFor(profileName)
    val profiled: ProfiledPlan = JvmRandomAccessSource.open(file).use { src ->
        val reader = StreamingGGUFReader.open(src)
        val stored = reader.planInput(ctx = ctx, prefillChunk = prefill, kvMode = kvMode)
        // Which encodings the *target* can feed is not a property of this machine, so it is asked
        // for rather than detected. `all` keeps the plan exactly as it was before #1116.
        val input = when {
            profile == null -> stored
            else -> try {
                stored.resolveWeightForms(
                    profile,
                    if (kernels == "dense") KernelCapabilities.DENSE_ONLY else KernelCapabilities.EVERYTHING,
                )
            } catch (e: IllegalStateException) {
                // A strict profile refuses a weight nothing on the target can feed. That is an
                // answer to the question the planner was asked, not a crash, so it reads as one.
                System.err.println("skainet plan: ${e.message}")
                exitProcess(1)
            }
        }
        val available = budget?.let { parseBytes(it) } ?: Runtime.getRuntime().maxMemory()
        if (profile != null) {
            // A profile owns the reserve, so --budget names what the *device* has, not what the
            // plan may use; without a profile the flag keeps its original meaning.
            profile.plan(input, available)
        } else {
            val b = when {
                noBudget -> null
                budget != null -> Budget.of(available)
                else -> Budget.available(available)
            }
            ProfiledPlan(PlannerProfile("none", reserveBytes = 0), MemoryPlans.plan(input, b), emptyList())
        }
    }
    print(if (profile != null) profiled.render() else profiled.plan.render())
    list?.let { glob -> print(renderList(profiled.plan, glob)) }
    exitProcess(if (profiled.plan.fits == false) 1 else 0)
}

/** The profile behind a `--profile` value; `null` for "none" (the plan's own defaults). */
internal fun profileFor(name: String): PlannerProfile? = when (name) {
    "mobile" -> PlannerProfile.MOBILE_2GB
    "desktop" -> PlannerProfile.DESKTOP
    "native" -> PlannerProfile.NATIVE
    else -> null
}

/** `1.3G`, `900M`, `64K`, `123456` → bytes (decimal suffixes are binary multiples, as the plan prints them). */
internal fun parseBytes(text: String): Long {
    val t = text.trim().uppercase().removeSuffix("B")
    val mult = when (t.lastOrNull()) { 'G' -> 1L shl 30; 'M' -> 1L shl 20; 'K' -> 1L shl 10; else -> 1L }
    val num = if (mult == 1L) t else t.dropLast(1)
    val value = num.toDoubleOrNull() ?: throw IllegalArgumentException("Not a size: '$text' (use e.g. 1.3G, 900M)")
    return (value * mult).toLong()
}

/** `model.layers[3].*` → regex; `*` matches anything, `?` one char, everything else literally. */
internal fun globToRegex(glob: String): Regex =
    Regex("^" + glob.split('*').joinToString(".*") { part -> part.split('?').joinToString(".") { Regex.escape(it) } } + "$")

internal fun renderList(plan: MemoryPlan, glob: String): String = buildString {
    val re = globToRegex(glob)
    val rows = plan.input.weights.filter { w -> (w.id?.canonical ?: w.name).let { re.matches(it) } }
    append('\n'); append("tensors matching '").append(glob).append("': ").append(rows.size).append('\n')
    val idWidth = (rows.maxOfOrNull { (it.id?.canonical ?: "—").length } ?: 10).coerceAtMost(60)
    for (w in rows) {
        append("  ")
        append((w.id?.canonical ?: "—").padEnd(idWidth)); append("  ")
        append(w.format.toString().padEnd(18))
        append("n=").append(w.elementCount.toString().padEnd(12))
        append(MemoryPlans.formatBytes(w.bytes).padStart(8))
        append("   ← ").append(w.name)
        append('\n')
    }
}
