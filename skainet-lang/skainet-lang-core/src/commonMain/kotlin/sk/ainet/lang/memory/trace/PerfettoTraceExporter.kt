package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.ScopeKind

/**
 * Renders a [TraceEvent] stream as a Chrome/Perfetto trace (the JSON array format Perfetto and
 * `chrome://tracing` both read) — SKEEP-003 §4.9, PRD M1-F7 / M1-A7.
 *
 * The mapping is the one the design asks for:
 * - **one track per scope** — phases and kernels on the main thread, allocations on a thread named
 *   after their [ScopeKind], so `Forward` and `Model` allocations are visually separate;
 * - **kernel runs and phases as duration slices**, labelled by op and `TensorId`s;
 * - **adapter insertions as instant events** with their byte cost, so a silent dequantisation shows
 *   up as a mark on the timeline (the #782 class);
 * - **live bytes per scope as counter tracks**, which is what makes a flat decode loop *look* flat;
 * - the memory plan as metadata on the trace.
 *
 * Timestamps are microseconds (Perfetto's unit) derived from the events' nanosecond clock.
 */
public object PerfettoTraceExporter {

    private const val PID: Int = 1
    private const val MAIN_TID: Int = 1

    /** Render [events] as a Chrome trace JSON document. */
    public fun export(events: List<TraceEvent>, processName: String = "skainet"): String {
        val out = StringBuilder(1024)
        out.append("{\"traceEvents\":[\n")
        var first = true
        fun emit(json: String) {
            if (!first) out.append(",\n")
            out.append(json); first = false
        }

        emit(metadata(PID, MAIN_TID, "process_name", processName))
        emit(metadata(PID, MAIN_TID, "thread_name", "phases & kernels"))
        val scopeTids = HashMap<ScopeKind, Int>()
        fun tidOf(scope: ScopeKind): Int = scopeTids.getOrPut(scope) {
            val tid = MAIN_TID + 1 + scopeTids.size
            emit(metadata(PID, tid, "thread_name", "${scope.name.lowercase()} scope"))
            tid
        }

        // live bytes per scope, updated as allocations come and go
        val live = HashMap<ScopeKind, Long>()

        for (e in events) {
            val ts = e.timeNanos / 1000.0
            when (e) {
                is TraceEvent.PhaseBegin -> emit(slice("B", e.phase + (e.step?.let { "#$it" } ?: ""), "phase", ts, MAIN_TID, e.attributes))
                is TraceEvent.PhaseEnd -> emit(slice("E", e.phase + (e.step?.let { "#$it" } ?: ""), "phase", ts, MAIN_TID, emptyMap()))
                is TraceEvent.KernelRun -> {
                    val args = buildMap {
                        put("kernel", e.kernel)
                        put("bytesRead", e.bytesRead.toString())
                        put("bytesWritten", e.bytesWritten.toString())
                        e.inputs.forEachIndexed { i, id -> if (id != null) put("in$i", id.canonical) }
                        e.output?.let { put("out", it.canonical) }
                    }
                    emit(complete(e.op, "kernel", (e.timeNanos - e.durationNanos) / 1000.0, e.durationNanos / 1000.0, MAIN_TID, args))
                }
                is TraceEvent.AdapterInserted -> emit(
                    instant(
                        "adapter:${e.kind}", "adapter", ts, MAIN_TID,
                        mapOf(
                            "from" to e.from.toString(), "to" to e.to.toString(),
                            "bytes" to e.bytes.toString(), "bytesBefore" to e.bytesBefore.toString(),
                            "bytesDelta" to e.bytesDelta.toString(),
                            "target" to (e.target?.canonical ?: "—"),
                        ),
                    ),
                )
                is TraceEvent.Allocation -> {
                    val now = (live[e.scope] ?: 0L) + e.bytes
                    live[e.scope] = now
                    emit(instant("alloc #${e.storageId}", "alloc", ts, tidOf(e.scope), mapOf("bytes" to e.bytes.toString(), "origin" to (e.origin?.canonical ?: "—"), "site" to (e.site ?: "—"))))
                    emit(counter("live bytes", ts, mapOf(e.scope.name.lowercase() to now)))
                }
                is TraceEvent.Free -> {
                    val now = ((live[e.scope] ?: 0L) - e.bytes).coerceAtLeast(0L)
                    live[e.scope] = now
                    emit(instant("free #${e.storageId}", "alloc", ts, tidOf(e.scope), mapOf("bytes" to e.bytes.toString())))
                    emit(counter("live bytes", ts, mapOf(e.scope.name.lowercase() to now)))
                }
                is TraceEvent.ScopeReset -> {
                    live[e.scope] = e.liveBytesAfter
                    emit(instant("reset", "scope", ts, tidOf(e.scope), mapOf("before" to e.liveBytesBefore.toString(), "after" to e.liveBytesAfter.toString())))
                    emit(counter("live bytes", ts, mapOf(e.scope.name.lowercase() to e.liveBytesAfter)))
                }
                is TraceEvent.Counter -> emit(counter(e.name, ts, mapOf(e.unit to e.value)))
                is TraceEvent.ScheduleDowngraded -> emit(
                    instant("schedule downgraded", "schedule", ts, MAIN_TID, mapOf("requested" to e.requested, "effective" to e.effective, "reason" to e.reason)),
                )
                is TraceEvent.ScheduleRegion -> emit(
                    complete("schedule ${e.op}", "schedule", (e.timeNanos - e.durationNanos) / 1000.0, e.durationNanos / 1000.0, MAIN_TID,
                        mapOf("schedule" to e.schedule, "elements" to e.elements.toString(), "tasks" to e.tasks.toString())),
                )
                is TraceEvent.Plan -> emit(
                    instant(
                        "plan", "plan", ts, MAIN_TID,
                        mapOf(
                            "model" to e.model, "ctx" to e.ctx.toString(),
                            "weights" to e.weightsBytes.toString(), "kv" to e.kvBytes.toString(),
                            "forward" to e.forwardBytes.toString(), "headroom" to e.headroomBytes.toString(),
                            "total" to e.totalBytes.toString(), "budget" to (e.budgetBytes?.toString() ?: "—"),
                            "fits" to (e.fits?.toString() ?: "—"),
                        ),
                    ),
                )
            }
        }
        out.append("\n],\"displayTimeUnit\":\"ms\"}")
        return out.toString()
    }

    /** Export the events a [RecordingTraceSink] has kept. */
    public fun export(sink: RecordingTraceSink, processName: String = "skainet"): String = export(sink.events(), processName)

    // --- Chrome trace event objects ---

    private fun slice(phase: String, name: String, cat: String, ts: Double, tid: Int, args: Map<String, String>): String =
        """{"ph":"$phase","name":"${esc(name)}","cat":"$cat","pid":$PID,"tid":$tid,"ts":${fmt(ts)}${argsJson(args)}}"""

    private fun complete(name: String, cat: String, ts: Double, dur: Double, tid: Int, args: Map<String, String>): String =
        """{"ph":"X","name":"${esc(name)}","cat":"$cat","pid":$PID,"tid":$tid,"ts":${fmt(ts)},"dur":${fmt(dur)}${argsJson(args)}}"""

    private fun instant(name: String, cat: String, ts: Double, tid: Int, args: Map<String, String>): String =
        """{"ph":"i","name":"${esc(name)}","cat":"$cat","pid":$PID,"tid":$tid,"ts":${fmt(ts)},"s":"t"${argsJson(args)}}"""

    private fun counter(name: String, ts: Double, values: Map<String, Long>): String =
        """{"ph":"C","name":"${esc(name)}","pid":$PID,"tid":$MAIN_TID,"ts":${fmt(ts)},"args":{${values.entries.joinToString(",") { "\"${esc(it.key)}\":${it.value}" }}}}"""

    private fun metadata(pid: Int, tid: Int, name: String, value: String): String =
        """{"ph":"M","name":"$name","pid":$pid,"tid":$tid,"args":{"name":"${esc(value)}"}}"""

    private fun argsJson(args: Map<String, String>): String =
        if (args.isEmpty()) "" else ",\"args\":{${args.entries.joinToString(",") { "\"${esc(it.key)}\":\"${esc(it.value)}\"" }}}"

    /** Microseconds with three decimals, without depending on a platform formatter. */
    private fun fmt(v: Double): String {
        val scaled = kotlin.math.round(v * 1000).toLong()
        val whole = scaled / 1000
        val frac = (if (scaled < 0) -scaled else scaled) % 1000
        return "$whole.${frac.toString().padStart(3, '0')}"
    }

    private fun esc(s: String): String = buildString(s.length) {
        for (c in s) when (c) {
            '"' -> append("\\\""); '\\' -> append("\\\\"); '\n' -> append("\\n"); '\r' -> append("\\r"); '\t' -> append("\\t")
            else -> if (c < ' ') append("\\u").append(c.code.toString(16).padStart(4, '0')) else append(c)
        }
    }
}
