package sk.ainet.lang.memory.trace

import sk.ainet.lang.tensor.TensorId

/**
 * The phase vocabulary of a generation loop (SKEEP-003 §4.9). Producers open spans with the
 * helpers below; [GenerationMetrics] reads them back. Both sides use these constants, so a typo
 * cannot silently produce a metric of zero.
 */
public object Phases {
    /** Weights being read off disk / mapped. */
    public const val LOAD: String = "load"
    /** Graph preparation before the first token. */
    public const val COMPILE: String = "compile"
    /** The prompt pass; its `tokens` attribute carries the prompt length. */
    public const val PREFILL: String = "prefill"
    /** One decode step; its `step` is the token index, starting at 1. */
    public const val DECODE: String = "decode"
    /** Turning logits into a token. */
    public const val SAMPLE: String = "sample"

    /** Attribute marking a span as a module rather than a generation phase. */
    public const val ATTR_KIND: String = "kind"
    public const val KIND_MODULE: String = "module"
    /** Attribute carrying the prompt length of a [PREFILL] span. */
    public const val ATTR_TOKENS: String = "tokens"
}

/** Counter names the metrics reader understands (`TraceEvent.Counter`). */
public object Counters {
    /** Resident set size, bytes. */
    public const val RSS: String = "rss"
    /** Cumulative major page faults — the number that must stay flat on mapped weights (M2-A4). */
    public const val PAGE_FAULTS: String = "page faults"
    /** Derived: tokens per second during decode. */
    public const val DECODE_TOKENS_PER_SECOND: String = "decode tok/s"
    /** Derived: tokens per second during prefill. */
    public const val PREFILL_TOKENS_PER_SECOND: String = "prefill tok/s"
    /** Derived: bytes read per second during decode. */
    public const val EFFECTIVE_BANDWIDTH: String = "effective bandwidth"
    /** Derived: [EFFECTIVE_BANDWIDTH] as a percentage of the device's peak. */
    public const val BANDWIDTH_UTILIZATION: String = "bandwidth utilization"
    /** Derived: time to first token, microseconds. */
    public const val TIME_TO_FIRST_TOKEN: String = "time to first token"
}

/** The prompt pass over [tokens] tokens. */
public inline fun <T> TraceSink.prefill(tokens: Int, block: () -> T): T =
    phase(Phases.PREFILL, attributes = mapOf(Phases.ATTR_TOKENS to tokens.toString()), block = block)

/** One decode step; [step] is the token index (1-based). */
public inline fun <T> TraceSink.decodeStep(step: Int, block: () -> T): T =
    phase(Phases.DECODE, step, block = block)

/** The sampling that turns this step's logits into a token. */
public inline fun <T> TraceSink.sample(step: Int? = null, block: () -> T): T =
    phase(Phases.SAMPLE, step, block = block)

/**
 * A module span nested inside the current phase — `model.layers[3].attn`, `model.lm_head`. These
 * are what [GenerationMetrics.modules] aggregates into the per-layer breakdown.
 */
public inline fun <T> TraceSink.module(path: String, step: Int? = null, block: () -> T): T =
    phase(path, step, mapOf(Phases.ATTR_KIND to Phases.KIND_MODULE), block)

/** A module span named after [id]'s module path. */
public inline fun <T> TraceSink.module(id: TensorId, step: Int? = null, block: () -> T): T =
    module(id.modulePath.joinToString("."), step, block)

/** Record a counter sample (RSS, page faults, a derived metric). */
public fun TraceSink.counter(name: String, value: Long, unit: String = "bytes") {
    if (isEnabled) emit(TraceEvent.Counter(name, value, unit))
}
