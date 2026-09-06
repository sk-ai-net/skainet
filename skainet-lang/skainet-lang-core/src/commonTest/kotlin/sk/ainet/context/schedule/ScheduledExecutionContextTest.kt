package sk.ainet.context.schedule

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.ScheduledExecutionContext
import sk.ainet.context.forwardScope
import sk.ainet.context.withSchedule
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-005: a schedule is a deployment property on the context. The default is sequential, a
 * context that cannot rebuild its ops reports an unhonoured request as a trace event (never a
 * silent downgrade), and the decorator survives the other decorators it composes with.
 */
class ScheduledExecutionContextTest {

    /** A schedule that only counts; enough to prove which one a context carries. */
    private class Counting(override val parallelism: Int = 4) : Schedule {
        var regions = 0
        override val name: String get() = "counting($parallelism)"
        override fun forRange(n: Int, grain: Int, body: (Int, Int) -> Unit) {
            regions++
            Schedule.Sequential.forRange(n, grain, body)
        }
    }

    /** Delegation would forward `withSchedule` to the delegate (and its Noop sink); route it through the interface default. */
    private class TracingContext(sink: TraceSink) : ExecutionContext by DefaultDataExecutionContext() {
        override val traceSink: TraceSink = sink
        override fun withSchedule(schedule: Schedule): ExecutionContext = super<ExecutionContext>.withSchedule(schedule)
    }

    @Test
    fun defaultScheduleIsSequential() {
        val ctx = DefaultDataExecutionContext()
        assertSame(Schedule.Sequential, ctx.schedule)
        assertEquals(1, ctx.schedule.parallelism)
        assertEquals("sequential", ctx.schedule.name)
    }

    @Test
    fun sequentialRunsTheWholeRangeOnce() {
        val seen = mutableListOf<Pair<Int, Int>>()
        Schedule.Sequential.forRange(7, grain = 3) { s, e -> seen += s to e }
        assertEquals(listOf(0 to 7), seen)
        Schedule.Sequential.forRange(0) { _, _ -> error("empty range must not call the body") }
        var count = 0
        Schedule.Sequential.forEach(5) { count += it }
        assertEquals(0 + 1 + 2 + 3 + 4, count)
    }

    @Test
    fun tasksForNeverExceedsElementsOrParallelism() {
        assertEquals(0, Schedule.tasksFor(0, 1, 8))
        assertEquals(1, Schedule.tasksFor(1, 1, 8))
        assertEquals(4, Schedule.tasksFor(4, 1, 8))
        assertEquals(8, Schedule.tasksFor(1000, 1, 8))
        assertEquals(3, Schedule.tasksFor(100, 40, 8), "grain caps the task count")
        assertEquals(1, Schedule.tasksFor(100, 1, 0), "parallelism is coerced to at least one")
    }

    @Test
    fun decoratorCarriesTheScheduleAndReportsAnUnhonouredRequest() {
        val sink = RecordingTraceSink()
        val requested = Counting()
        val scheduled = ScheduledExecutionContext(TracingContext(sink), requested)

        assertSame(requested, scheduled.schedule, "the decorator answers with the requested schedule")
        val downgrades = sink.eventsOf<TraceEvent.ScheduleDowngraded>()
        assertEquals(1, downgrades.size, "a base that cannot rebuild its ops must say so")
        assertEquals("counting(4)", downgrades.single().requested)
        assertEquals("sequential", downgrades.single().effective)
    }

    @Test
    fun requestingTheScheduleAContextAlreadyRunsIsSilent() {
        val sink = RecordingTraceSink()
        val ctx = TracingContext(sink)
        assertSame(ctx, ctx.withSchedule(Schedule.Sequential))
        assertTrue(sink.eventsOf<TraceEvent.ScheduleDowngraded>().isEmpty())
    }

    @Test
    fun withScheduleBlockHandsOutADecoratedContext() {
        val counting = Counting()
        val result = DefaultDataExecutionContext().withSchedule(counting) { ctx ->
            assertSame(counting, ctx.schedule)
            ctx.schedule.forRange(10) { _, _ -> }
            "ok"
        }
        assertEquals("ok", result)
        assertEquals(1, counting.regions)
    }

    @Test
    fun scheduleSurvivesAForwardScope() {
        val counting = Counting()
        DefaultDataExecutionContext().withSchedule(counting) { scheduled ->
            scheduled.forwardScope(slabFloats = 16) { scoped, _ ->
                assertSame(counting, scoped.schedule, "forwardScope rebuilds through withTensorDataFactory; the schedule must be kept")
            }
        }
    }

    @Test
    fun scheduleHintRoundTripsThroughItsMapForm() {
        val hint = ScheduleHint.parallel("batch", "heads", parallelism = 8)
        val map = hint.toAttributeMap()
        assertEquals(hint, ScheduleHint.fromAttribute(map))
        assertEquals(hint, ScheduleHint.fromAttribute(hint))
        assertEquals(null, ScheduleHint.fromAttribute("nonsense"))
        assertEquals(null, ScheduleHint.fromAttribute(mapOf(ScheduleHint.DIMS_KEY to emptyList<String>())))
    }
}
