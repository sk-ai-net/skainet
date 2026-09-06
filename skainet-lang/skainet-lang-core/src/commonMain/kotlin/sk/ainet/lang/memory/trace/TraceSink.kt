package sk.ainet.lang.memory.trace


/**
 * Where [TraceEvent]s go. Disabled by default ([NoopTraceSink]) — emitting costs one `isEnabled`
 * check — and opt-in per `ExecutionContext` (`traceSink`). Implementations: [RecordingTraceSink]
 * (ring buffer, tests and the debugger), [CompositeTraceSink], and the exporters of M1 slice #1025
 * (Perfetto JSON, JFR, `android.os.Trace`).
 */
public interface TraceSink {
    /** `false` for a sink that drops everything; producers check it before building an event. */
    public val isEnabled: Boolean get() = true

    public fun emit(event: TraceEvent)
}

/** The default: nothing is recorded, nothing is allocated. */
public object NoopTraceSink : TraceSink {
    override val isEnabled: Boolean get() = false
    override fun emit(event: TraceEvent) {}
}

/**
 * Keeps the last [capacity] events in a ring buffer. The debugger, tests and the plan-vs-actual
 * check read [events]; [clear] starts over. Not thread-safe by design (one sink per context).
 */
public class RecordingTraceSink(public val capacity: Int = DEFAULT_CAPACITY) : TraceSink {
    init { require(capacity > 0) { "capacity must be > 0" } }

    private val buffer = ArrayDeque<TraceEvent>(minOf(capacity, 1024))
    /** Total number of events ever emitted (including ones that fell out of the ring). */
    public var emitted: Long = 0L
        private set
    /** Events dropped from the head because the ring was full. */
    public var dropped: Long = 0L
        private set

    override fun emit(event: TraceEvent) {
        if (buffer.size == capacity) { buffer.removeFirst(); dropped++ }
        buffer.addLast(event); emitted++
    }

    /** The retained events, oldest first. */
    public fun events(): List<TraceEvent> = buffer.toList()

    public inline fun <reified T : TraceEvent> eventsOf(): List<T> = events().filterIsInstance<T>()

    public fun clear() { buffer.clear(); dropped = 0L; emitted = 0L }

    public companion object { public const val DEFAULT_CAPACITY: Int = 65_536 }
}

/** Fan-out to several sinks; enabled if any of them is. */
public class CompositeTraceSink(private val sinks: List<TraceSink>) : TraceSink {
    public constructor(vararg sinks: TraceSink) : this(sinks.toList())
    override val isEnabled: Boolean get() = sinks.any { it.isEnabled }
    override fun emit(event: TraceEvent) { for (s in sinks) if (s.isEnabled) s.emit(event) }
}

/**
 * Run [block] inside a phase span: emits [TraceEvent.PhaseBegin], then [TraceEvent.PhaseEnd] with
 * the measured duration (also on exception). No events and no allocation when the sink is disabled.
 */
public inline fun <T> TraceSink.phase(name: String, step: Int? = null, attributes: Map<String, String> = emptyMap(), block: () -> T): T {
    if (!isEnabled) return block()
    val t0 = TraceClock.nowNanos()
    emit(TraceEvent.PhaseBegin(name, step, attributes, t0))
    try {
        return block()
    } finally {
        val t1 = TraceClock.nowNanos()
        emit(TraceEvent.PhaseEnd(name, step, t1 - t0, t1))
    }
}

/** Time [block] as a kernel run of [op] on [kernel]; the returned value is the kernel's result. */
public inline fun <T> TraceSink.kernel(
    op: String,
    kernel: String,
    inputs: List<sk.ainet.lang.tensor.TensorId?> = emptyList(),
    output: sk.ainet.lang.tensor.TensorId? = null,
    bytesRead: Long = 0L,
    bytesWritten: Long = 0L,
    block: () -> T,
): T {
    if (!isEnabled) return block()
    val t0 = TraceClock.nowNanos()
    val r = block()
    val t1 = TraceClock.nowNanos()
    emit(TraceEvent.KernelRun(op, kernel, inputs, output, bytesRead, bytesWritten, t1 - t0, t1))
    return r
}
