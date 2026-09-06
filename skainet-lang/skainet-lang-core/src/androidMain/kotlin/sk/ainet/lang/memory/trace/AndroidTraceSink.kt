package sk.ainet.lang.memory.trace

import android.os.Trace

/**
 * A [TraceSink] that forwards SKaiNET's events to `android.os.Trace`, so a memory-architecture
 * trace lines up with the OS view in Perfetto / systrace on device (SKEEP-003 §4.9, decision #12).
 *
 * Phases and kernel runs become async sections (they can nest and overlap across steps); adapters
 * and scope resets become instants; allocation totals become counters, which is what makes a flat
 * decode loop visible next to the app's own memory graph.
 *
 * Section names are truncated to `android.os.Trace`'s limit (127 characters), and tracing is only
 * enabled while the OS is capturing, so a shipping build pays one boolean check per event.
 */
public class AndroidTraceSink : TraceSink {

    override val isEnabled: Boolean get() = Trace.isEnabled()

    override fun emit(event: TraceEvent) {
        if (!Trace.isEnabled()) return
        when (event) {
            is TraceEvent.PhaseBegin -> Trace.beginAsyncSection(name(event.phase, event.step), cookie(event.phase, event.step))
            is TraceEvent.PhaseEnd -> Trace.endAsyncSection(name(event.phase, event.step), cookie(event.phase, event.step))
            is TraceEvent.KernelRun -> {
                // the kernel already ran; represent it as a zero-length async pair so it shows on the timeline
                val n = truncate("kernel ${event.op}:${event.kernel}")
                val c = (event.output?.canonical ?: event.kernel).hashCode()
                Trace.beginAsyncSection(n, c); Trace.endAsyncSection(n, c)
            }
            is TraceEvent.AdapterInserted -> Trace.setCounter(truncate("skainet adapter ${event.kind} bytes"), event.bytes)
            is TraceEvent.Allocation -> Trace.setCounter(truncate("skainet ${event.scope.name.lowercase()} alloc bytes"), event.bytes)
            is TraceEvent.Free -> Trace.setCounter(truncate("skainet ${event.scope.name.lowercase()} free bytes"), event.bytes)
            is TraceEvent.ScopeReset -> Trace.setCounter(truncate("skainet ${event.scope.name.lowercase()} live bytes"), event.liveBytesAfter)
            is TraceEvent.Counter -> Trace.setCounter(truncate("skainet ${event.name}"), event.value)
            is TraceEvent.ScheduleDowngraded -> Trace.setCounter(truncate("skainet schedule downgraded"), 1L)
            is TraceEvent.ScheduleRegion -> Trace.setCounter(truncate("skainet schedule ${event.op} tasks"), event.tasks.toLong())
            is TraceEvent.Plan -> Trace.setCounter(truncate("skainet plan total bytes"), event.totalBytes)
        }
    }

    private fun name(phase: String, step: Int?): String = truncate(if (step == null) "skainet $phase" else "skainet $phase#$step")
    private fun cookie(phase: String, step: Int?): Int = phase.hashCode() * 31 + (step ?: 0)

    /** `android.os.Trace` rejects names longer than 127 characters. */
    private fun truncate(s: String): String = if (s.length <= MAX_SECTION_NAME) s else s.substring(0, MAX_SECTION_NAME)

    private companion object { const val MAX_SECTION_NAME: Int = 127 }
}
