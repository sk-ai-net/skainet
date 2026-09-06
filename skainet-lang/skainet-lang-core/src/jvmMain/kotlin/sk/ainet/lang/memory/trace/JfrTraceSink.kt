package sk.ainet.lang.memory.trace

import jdk.jfr.Category
import jdk.jfr.Event
import jdk.jfr.EventFactory
import jdk.jfr.Label
import jdk.jfr.Name

/**
 * A [TraceSink] that emits SKaiNET's events as **JFR** events, so a memory-architecture trace opens
 * in the IntelliJ profiler or JDK Mission Control next to JIT, GC and allocation data (SKEEP-003
 * §4.9, decision #12).
 *
 * Recording is JFR's business: start the JVM with `-XX:StartFlightRecording=...` (or use
 * `jcmd JFR.start`) and the events appear under the **SKaiNET** category. Emission is cheap and
 * self-disabling — JFR drops an event whose type is not enabled, and [isEnabled] mirrors that.
 */
public class JfrTraceSink : TraceSink {

    @Name("sk.ainet.KernelRun") @Label("SKaiNET kernel run") @Category("SKaiNET")
    internal class KernelRunEvent : Event() {
        @Label("Op") var op: String? = null
        @Label("Kernel") var kernel: String? = null
        @Label("Output tensor") var output: String? = null
        @Label("Bytes read") var bytesRead: Long = 0
        @Label("Bytes written") var bytesWritten: Long = 0
    }

    @Name("sk.ainet.Phase") @Label("SKaiNET phase") @Category("SKaiNET")
    internal class PhaseEvent : Event() {
        @Label("Phase") var phase: String? = null
        @Label("Step") var step: Int = -1
        @Label("Duration (ns)") var durationNanos: Long = 0
    }

    @Name("sk.ainet.Allocation") @Label("SKaiNET allocation") @Category("SKaiNET")
    internal class AllocationEvent : Event() {
        @Label("Storage id") var storageId: Long = 0
        @Label("Scope") var scope: String? = null
        @Label("Bytes") var bytes: Long = 0
        @Label("Origin") var origin: String? = null
        @Label("Freed") var freed: Boolean = false
    }

    @Name("sk.ainet.Adapter") @Label("SKaiNET adapter inserted") @Category("SKaiNET")
    internal class AdapterEvent : Event() {
        @Label("Kind") var kind: String? = null
        @Label("From") var from: String? = null
        @Label("To") var to: String? = null
        @Label("Bytes") var bytes: Long = 0
        @Label("Bytes before") var bytesBefore: Long = 0
        @Label("Target") var target: String? = null
    }

    override val isEnabled: Boolean
        get() = KernelRunEvent().isEnabled || AllocationEvent().isEnabled || PhaseEvent().isEnabled || AdapterEvent().isEnabled

    override fun emit(event: TraceEvent) {
        when (event) {
            is TraceEvent.KernelRun -> KernelRunEvent().apply {
                op = event.op; kernel = event.kernel; output = event.output?.canonical
                bytesRead = event.bytesRead; bytesWritten = event.bytesWritten
            }.commit()
            is TraceEvent.PhaseEnd -> PhaseEvent().apply {
                phase = event.phase; step = event.step ?: -1; durationNanos = event.durationNanos
            }.commit()
            is TraceEvent.Allocation -> AllocationEvent().apply {
                storageId = event.storageId; scope = event.scope.name; bytes = event.bytes; origin = event.origin?.canonical; freed = false
            }.commit()
            is TraceEvent.Free -> AllocationEvent().apply {
                storageId = event.storageId; scope = event.scope.name; bytes = event.bytes; freed = true
            }.commit()
            is TraceEvent.AdapterInserted -> AdapterEvent().apply {
                kind = event.kind; from = event.from.toString(); to = event.to.toString()
                bytes = event.bytes; bytesBefore = event.bytesBefore; target = event.target?.canonical
            }.commit()
            // PhaseBegin is covered by PhaseEnd's duration; counters/plans have no JFR analogue yet.
            else -> Unit
        }
    }

    /** JFR's own view of whether anything is recording (used by tests and diagnostics). */
    public fun anyEventEnabled(): Boolean = isEnabled

    private companion object {
        // touch EventFactory so the class is linked eagerly and a missing jdk.jfr module fails fast
        @Suppress("unused") private val JFR_PRESENT: Boolean = EventFactory::class.java.name.isNotEmpty()
    }
}
