package sk.ainet.exec.schedule

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.asExecutor
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceClock
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * The JVM [Schedule] (SKEEP-005): a region is a queue of `tasks` chunks shared between the
 * calling thread and `tasks - 1` helpers dispatched to [dispatcher]. The caller runs the first
 * chunk itself, then keeps claiming chunks until the queue is empty, and only then waits — and
 * only for chunks a thread has already claimed. That is what makes the region safe to enter
 * from a thread of the dispatcher's own pool: a caller never blocks on work that still needs a
 * pool thread, because it does that work itself. A helper that the pool never got round to
 * running finds the queue empty and exits without touching the body.
 *
 * The contract's guarantees hold as before: the first failure stops every chunk that has not
 * started, is rethrown once every running chunk has finished (later failures are attached as
 * suppressed), no body outlives [forRange], and every chunk's writes happen-before the return.
 *
 * A region reached from inside another region (a body that calls into a parallel op despite the
 * contract) runs inline. Callers that want isolation from `Dispatchers.Default` use [dedicated].
 *
 * A previous version ran the region as `runBlocking { coroutineScope { launch(dispatcher) … } }`.
 * A `coroutineScope` waits for every child, including the ones the pool never started, so a
 * region entered from `Dispatchers.Default` deadlocked as soon as every worker was inside one —
 * routinely on a 4-vCPU CI runner, never on a 14-core laptop.
 */
public open class CoroutineSchedule @JvmOverloads constructor(
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    final override val parallelism: Int = Runtime.getRuntime().availableProcessors(),
    private val sink: TraceSink = NoopTraceSink,
    private val label: String = "coroutines",
) : Schedule {

    init {
        require(parallelism >= 1) { "CoroutineSchedule: parallelism must be >= 1, got $parallelism" }
    }

    final override val name: String = "$label($parallelism)"

    private val executor: Executor = dispatcher.asExecutor()

    override fun forRange(n: Int, grain: Int, body: (start: Int, end: Int) -> Unit) {
        val tasks = Schedule.tasksFor(n, grain, parallelism)
        if (tasks == 0) return
        if (tasks == 1 || inRegion.get() == true) {
            body(0, n)
            return
        }
        val chunk = Schedule.chunkFor(n, tasks)
        val started = if (sink.isEnabled) TraceClock.nowNanos() else 0L
        val region = Region(n, chunk, tasks, body)
        inRegion.set(true)
        try {
            repeat(tasks - 1) { executor.execute(region) }
            region.runChunk(0)
            region.drain()
        } finally {
            inRegion.set(false)
        }
        region.awaitClaimed()
        region.rethrow()
        if (sink.isEnabled) {
            val now = TraceClock.nowNanos()
            sink.emit(TraceEvent.ScheduleRegion(op = "forRange", schedule = name, elements = n, tasks = tasks, durationNanos = now - started, timeNanos = now))
        }
    }

    /** One `forRange` call: the chunk queue, the completion latch and the first failure. */
    private class Region(
        private val n: Int,
        private val chunk: Int,
        private val tasks: Int,
        private val body: (Int, Int) -> Unit,
    ) : Runnable {
        /** Next chunk to claim; the caller takes chunk 0 before the helpers start. */
        private val next = AtomicInteger(1)

        /** Counts down once per chunk, whether it ran or was skipped after a failure. */
        private val remaining = CountDownLatch(tasks)

        private val failure = AtomicReference<Throwable?>(null)

        /** Helper entry point: run on a pool thread, claim chunks until none are left. */
        override fun run() {
            val previous = inRegion.get()
            inRegion.set(true)
            try {
                drain()
            } finally {
                inRegion.set(previous)
            }
        }

        fun drain() {
            while (true) {
                val i = next.getAndIncrement()
                if (i >= tasks) return
                runChunk(i)
            }
        }

        fun runChunk(i: Int) {
            try {
                if (failure.get() == null) body(i * chunk, minOf((i + 1) * chunk, n))
            } catch (t: Throwable) {
                if (!failure.compareAndSet(null, t)) failure.get()!!.addSuppressed(t)
            } finally {
                remaining.countDown()
            }
        }

        /**
         * Waits for the chunks other threads have claimed. By the time the caller gets here it
         * has drained the queue, so every outstanding chunk is on a thread that is running it.
         */
        fun awaitClaimed() {
            var interrupted = false
            while (true) {
                try {
                    remaining.await()
                    break
                } catch (_: InterruptedException) {
                    interrupted = true
                }
            }
            if (interrupted) Thread.currentThread().interrupt()
        }

        fun rethrow() {
            failure.get()?.let { throw it }
        }
    }

    override fun toString(): String = name

    public companion object {
        /** Set on any thread currently executing a region body, so a nested region runs inline. */
        private val inRegion: ThreadLocal<Boolean> = ThreadLocal()

        /** Core-count parallelism on `Dispatchers.Default` — the platform default schedule on the JVM. */
        @JvmStatic
        public fun hardware(sink: TraceSink = NoopTraceSink): CoroutineSchedule =
            CoroutineSchedule(Dispatchers.Default, Runtime.getRuntime().availableProcessors(), sink)

        /**
         * A schedule with its own daemon pool of `parallelism - 1` workers (the caller is the last
         * worker), for code that wants isolation from `Dispatchers.Default`. Close it when done.
         */
        @JvmStatic
        @JvmOverloads
        public fun dedicated(
            parallelism: Int = Runtime.getRuntime().availableProcessors(),
            sink: TraceSink = NoopTraceSink,
        ): DedicatedCoroutineSchedule {
            require(parallelism >= 1) { "dedicated: parallelism must be >= 1, got $parallelism" }
            val counter = AtomicInteger()
            val executor = Executors.newFixedThreadPool(maxOf(1, parallelism - 1)) { r ->
                Thread(r, "skainet-schedule-${counter.incrementAndGet()}").apply { isDaemon = true }
            }
            return DedicatedCoroutineSchedule(executor, parallelism, sink)
        }
    }
}

/** [CoroutineSchedule] over an owned thread pool; [close] shuts the pool down. */
public class DedicatedCoroutineSchedule internal constructor(
    private val executor: ExecutorService,
    parallelism: Int,
    sink: TraceSink,
) : CoroutineSchedule(executor.asCoroutineDispatcher(), parallelism, sink, label = "dedicated"), AutoCloseable {
    override fun close() {
        executor.shutdown()
    }
}
