package sk.ainet.exec.schedule

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import java.util.BitSet
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-005: the JVM schedule honours the [Schedule.forRange] contract. */
@OptIn(ExperimentalMemoryApi::class)
class CoroutineScheduleTest {

    private fun coverage(schedule: Schedule, n: Int, grain: Int): Pair<BitSet, Int> {
        val seen = BitSet(n)
        val tasks = AtomicInteger()
        schedule.forRange(n, grain) { s, e ->
            tasks.incrementAndGet()
            synchronized(seen) {
                for (i in s until e) {
                    assertTrue(!seen[i], "index $i visited twice")
                    seen.set(i)
                }
            }
        }
        return seen to tasks.get()
    }

    @Test
    fun everyIndexIsVisitedExactlyOnceForAnyShape() {
        val schedule = CoroutineSchedule(parallelism = 4)
        for (n in listOf(0, 1, 7, 255, 1000, 4097)) {
            for (grain in listOf(1, 17, 10_000)) {
                val (seen, tasks) = coverage(schedule, n, grain)
                assertEquals(n, seen.cardinality(), "n=$n grain=$grain")
                assertEquals(Schedule.tasksFor(n, grain, 4), tasks, "n=$n grain=$grain task count")
            }
        }
    }

    @Test
    fun parallelismOneIsSequentialOnTheCallerThread() {
        val schedule = CoroutineSchedule(parallelism = 1)
        val caller = Thread.currentThread()
        val ranges = mutableListOf<Pair<Int, Int>>()
        schedule.forRange(100) { s, e ->
            assertSame(caller, Thread.currentThread())
            ranges += s to e
        }
        assertEquals(listOf(0 to 100), ranges)
        assertEquals("coroutines(1)", schedule.name)
    }

    @Test
    fun callerThreadRunsTheFirstChunkAndWorkersHelpWithTheRest() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val caller = Thread.currentThread()
        val onCaller = AtomicInteger()
        val elsewhere = AtomicInteger()
        var first: Thread? = null
        schedule.forRange(4000, grain = 1) { s, _ ->
            if (s == 0) first = Thread.currentThread()
            if (Thread.currentThread() === caller) onCaller.incrementAndGet() else elsewhere.incrementAndGet()
            Thread.sleep(30)   // long enough for the pool to claim the other chunks
        }
        assertSame(caller, first, "the caller runs the first chunk itself")
        assertTrue(elsewhere.get() >= 1, "at least one chunk ran on the dispatcher")
        assertEquals(4, onCaller.get() + elsewhere.get(), "every chunk ran exactly once")
    }

    @Test
    fun aFailingTaskCancelsSiblingsAndRethrowsAfterTheyFinish() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val running = AtomicInteger()
        val finished = AtomicInteger()
        val boom = assertFailsWith<IllegalStateException> {
            schedule.forRange(4, grain = 1) { s, _ ->
                running.incrementAndGet()
                try {
                    if (s == 2) error("task $s failed")
                    Thread.sleep(20)
                } finally {
                    finished.incrementAndGet()
                }
            }
        }
        assertEquals("task 2 failed", boom.message)
        assertEquals(running.get(), finished.get(), "no task may still be running when forRange returns")
    }

    @Test
    fun aNestedRegionRunsInlineOnTheWorkerThread() {
        val schedule = CoroutineSchedule(parallelism = 4)
        val nestedTasks = AtomicInteger()
        schedule.forRange(4, grain = 1) { _, _ ->
            val worker = Thread.currentThread()
            schedule.forRange(4000, grain = 1) { _, _ ->
                nestedTasks.incrementAndGet()
                assertSame(worker, Thread.currentThread(), "a nested region must not fork")
            }
        }
        assertEquals(4, nestedTasks.get(), "each outer task ran its nested region as one inline chunk")
    }

    @Test
    fun aRegionStartedFromADefaultDispatcherWorkerCompletes() {
        val schedule = CoroutineSchedule(parallelism = Runtime.getRuntime().availableProcessors())
        val total = AtomicInteger()
        runBlocking(Dispatchers.Default) {
            val jobs = List(4) {
                launch { schedule.forRange(4096, grain = 1) { s, e -> total.addAndGet(e - s) } }
            }
            jobs.forEach { it.join() }
        }
        assertEquals(4 * 4096, total.get())
    }

    /**
     * The CI deadlock: every thread of the schedule's own pool enters a region at once, so no
     * pool thread is free to run the helpers. The caller must finish the region by itself. With
     * the old `runBlocking { coroutineScope { … } }` region this hung forever on a pool of any
     * size, because the scope waited for children the pool could never start.
     */
    @Test
    fun aRegionEnteredFromEveryThreadOfItsOwnPoolStillCompletes() {
        val poolSize = 2
        val pool = Executors.newFixedThreadPool(poolSize)
        try {
            val schedule = CoroutineSchedule(dispatcher = pool.asCoroutineDispatcher(), parallelism = 4)
            val allInside = CyclicBarrier(poolSize)
            val total = AtomicInteger()
            val futures = List(poolSize) {
                pool.submit {
                    allInside.await(10, TimeUnit.SECONDS)
                    schedule.forRange(4096, grain = 1) { s, e -> total.addAndGet(e - s) }
                }
            }
            futures.forEach { it.get(30, TimeUnit.SECONDS) }
            assertEquals(poolSize * 4096, total.get())
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun dedicatedScheduleOwnsItsPoolAndCloses() {
        CoroutineSchedule.dedicated(parallelism = 3).use { schedule ->
            assertEquals("dedicated(3)", schedule.name)
            val (seen, _) = coverage(schedule, 999, 1)
            assertEquals(999, seen.cardinality())
        }
    }

    @Test
    fun regionsAreReportedToTheSink() {
        val sink = RecordingTraceSink()
        val schedule = CoroutineSchedule(parallelism = 2, sink = sink)
        schedule.forRange(10, grain = 1) { _, _ -> }
        schedule.forRange(1, grain = 1) { _, _ -> }   // single task: inline, no region event
        val regions = sink.eventsOf<TraceEvent.ScheduleRegion>()
        assertEquals(1, regions.size)
        assertEquals(10, regions.single().elements)
        assertEquals(2, regions.single().tasks)
        assertEquals("coroutines(2)", regions.single().schedule)
    }
}
