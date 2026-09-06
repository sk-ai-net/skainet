package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.Counters
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.memory.trace.counter

/**
 * What the *operating system* thinks this process is using (SKEEP-003 §4.9; M2-A1/M2-A3).
 *
 * The allocation events say what SKaiNET asked for; this says what the process actually holds and
 * how often it had to go to disk for it. Both are needed: a plan that matches the allocation
 * events and a resident set that keeps growing means the bytes are escaping somewhere the model
 * does not see, and mapped weights are only "free" while the page-fault rate stays near zero.
 *
 * Every value is `null` where the platform cannot answer — a browser has no `/proc`, and neither
 * has a Wasm host. Callers report "—", they do not guess.
 */
public expect object MemoryProbe {
    /** Resident set size in bytes, or `null` when the platform cannot say. */
    public fun rssBytes(): Long?

    /** Major page faults since process start — the ones that went to disk. */
    public fun majorFaults(): Long?

    /** Minor page faults since process start — the ones satisfied from page cache. */
    public fun minorFaults(): Long?
}

/** One sample of the process-level counters, with the fields the platform could answer. */
public data class ProcessMemorySample(
    val rssBytes: Long?,
    val majorFaults: Long?,
    val minorFaults: Long?,
) {
    /** Emit what is known as trace counters, so it lands beside the plan in an exported trace. */
    public fun emitTo(sink: TraceSink) {
        if (!sink.isEnabled) return
        rssBytes?.let { sink.counter(Counters.RSS, it) }
        majorFaults?.let { sink.counter(Counters.PAGE_FAULTS, it, unit = "faults") }
    }

    /** Major faults between this sample and an [earlier] one, or `null` if either is unknown. */
    public fun majorFaultsSince(earlier: ProcessMemorySample): Long? {
        val now = majorFaults ?: return null
        val then = earlier.majorFaults ?: return null
        return now - then
    }

    override fun toString(): String = buildString {
        append("rss=").append(rssBytes?.let { "${it / (1024 * 1024)} MB" } ?: "—")
        append(" majflt=").append(majorFaults?.toString() ?: "—")
        append(" minflt=").append(minorFaults?.toString() ?: "—")
    }
}

/** Sample all three counters at once. */
public fun MemoryProbe.sample(): ProcessMemorySample =
    ProcessMemorySample(rssBytes(), majorFaults(), minorFaults())
