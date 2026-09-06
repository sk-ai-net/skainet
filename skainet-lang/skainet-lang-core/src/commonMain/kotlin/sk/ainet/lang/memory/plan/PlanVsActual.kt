package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent

/**
 * What actually happened, reconstructed from the allocation events of a run (SKEEP-003 §4.9
 * "plan-vs-actual", PRD M1-F8 / M1-A8): peak live bytes per scope, plus the totals the plan
 * predicted. The point is to keep the planner honest as kernels change — a plan that drifts from
 * reality is worse than no plan, so CI compares them and fails past a threshold.
 */
public data class ActualMemory(
    /** Peak live bytes per scope over the run. */
    val peakByScope: Map<ScopeKind, Long>,
    /** Bytes allocated in total (including bytes later freed) per scope — the churn. */
    val allocatedByScope: Map<ScopeKind, Long>,
    /** Number of allocation events per scope; a decode loop should show none in `FORWARD` after warm-up. */
    val allocationsByScope: Map<ScopeKind, Int>,
    /** Bytes of adapters the dispatcher inserted (the #782 class), by kind. */
    val adapterBytesByKind: Map<String, Long>,
) {
    val peakModelBytes: Long get() = peakByScope[ScopeKind.MODEL] ?: 0L
    val peakForwardBytes: Long get() = peakByScope[ScopeKind.FORWARD] ?: 0L
    val peakTotalBytes: Long get() = peakByScope.values.sum()
    val adapterBytes: Long get() = adapterBytesByKind.values.sum()

    public companion object {
        /** Replay [events] and track live bytes per scope. */
        public fun from(events: List<TraceEvent>): ActualMemory {
            val live = HashMap<ScopeKind, Long>()
            val peak = HashMap<ScopeKind, Long>()
            val allocated = HashMap<ScopeKind, Long>()
            val counts = HashMap<ScopeKind, Int>()
            val adapters = HashMap<String, Long>()
            fun bump(scope: ScopeKind, delta: Long) {
                val now = ((live[scope] ?: 0L) + delta).coerceAtLeast(0L)
                live[scope] = now
                if (now > (peak[scope] ?: 0L)) peak[scope] = now
            }
            for (e in events) when (e) {
                is TraceEvent.Allocation -> {
                    bump(e.scope, e.bytes)
                    allocated[e.scope] = (allocated[e.scope] ?: 0L) + e.bytes
                    counts[e.scope] = (counts[e.scope] ?: 0) + 1
                }
                is TraceEvent.Free -> bump(e.scope, -e.bytes)
                is TraceEvent.ScopeReset -> { live[e.scope] = e.liveBytesAfter }
                is TraceEvent.AdapterInserted -> adapters[e.kind] = (adapters[e.kind] ?: 0L) + e.bytes
                else -> Unit
            }
            return ActualMemory(peak.toMap(), allocated.toMap(), counts.toMap(), adapters.toMap())
        }

        /** Replay what a [RecordingTraceSink] kept. */
        public fun from(sink: RecordingTraceSink): ActualMemory = from(sink.events())
    }
}

/** One line of the comparison: what the plan said, what the run did, and by how much they differ. */
public data class PlanVsActualLine(val section: String, val plannedBytes: Long, val actualBytes: Long) {
    /** Signed relative difference (`actual/planned - 1`); `null` when nothing was planned. */
    val relativeDrift: Double? get() = if (plannedBytes == 0L) null else (actualBytes - plannedBytes).toDouble() / plannedBytes
    val absoluteDrift: Long get() = actualBytes - plannedBytes
    /** Whether this line is within [tolerance] (a fraction, e.g. 0.10 for 10 %). */
    public fun withinTolerance(tolerance: Double): Boolean {
        val d = relativeDrift ?: return actualBytes == 0L
        return kotlin.math.abs(d) <= tolerance
    }
}

/**
 * The comparison itself (PRD M1-F8): the plan's resident sections against the peaks a run reached.
 * `check()` is what a CI acceptance run calls — a drift beyond the tolerance fails, which keeps the
 * planner honest as kernels change.
 */
public data class PlanVsActual(
    val plan: MemoryPlan,
    val actual: ActualMemory,
    val tolerance: Double = DEFAULT_TOLERANCE,
) {
    val lines: List<PlanVsActualLine> = listOf(
        PlanVsActualLine("weights (model scope)", plan.weightsBytes + plan.kvBytes, actual.peakModelBytes),
        PlanVsActualLine("forward slab", plan.forwardBytes, actual.peakForwardBytes),
    )

    /** Lines that drifted beyond [tolerance]. */
    public fun violations(): List<PlanVsActualLine> = lines.filterNot { it.withinTolerance(tolerance) }

    /** `true` when every line is within tolerance. */
    public val withinTolerance: Boolean get() = violations().isEmpty()

    /**
     * Throw when any line drifted beyond [tolerance] — the CI assertion of M1-F8. The message
     * carries the table, so a failure explains itself without a rerun.
     */
    public fun check() {
        if (withinTolerance) return
        throw IllegalStateException("Memory plan drifted from the actual run (tolerance ${(tolerance * 100).toInt()} %):\n" + render())
    }

    /** The comparison table. */
    public fun render(): String = buildString {
        append(plan.input.modelName); append(" · ctx "); append(plan.input.ctx); append(" · plan vs actual\n")
        append("  section                     planned      actual      drift\n")
        for (l in lines) {
            append("  "); append(l.section.padEnd(24))
            append(MemoryPlans.formatBytes(l.plannedBytes).padStart(10))
            append(MemoryPlans.formatBytes(l.actualBytes).padStart(12))
            val d = l.relativeDrift
            append((if (d == null) "—" else (if (d >= 0) "+" else "") + (d * 100).toInt().toString() + " %").padStart(11))
            if (!l.withinTolerance(tolerance)) append("   ✘")
            append('\n')
        }
        val fwdAllocs = actual.allocationsByScope[ScopeKind.FORWARD] ?: 0
        append("  forward-scope allocations: "); append(fwdAllocs)
        append(" (slab + overflow; steady-state decode should add none)\n")
        if (actual.adapterBytes > 0) {
            append("  adapters: "); append(MemoryPlans.formatBytes(actual.adapterBytes))
            append(" — "); append(actual.adapterBytesByKind.entries.joinToString(", ") { "${it.key} ${MemoryPlans.formatBytes(it.value)}" }); append('\n')
        }
    }

    public companion object {
        /** PRD M1-F8: a difference above 10 % fails the CI acceptance run. */
        public const val DEFAULT_TOLERANCE: Double = 0.10

        /** Compare [plan] with the run [sink] recorded. */
        public fun of(plan: MemoryPlan, sink: RecordingTraceSink, tolerance: Double = DEFAULT_TOLERANCE): PlanVsActual =
            PlanVsActual(plan, ActualMemory.from(sink), tolerance)
    }
}
