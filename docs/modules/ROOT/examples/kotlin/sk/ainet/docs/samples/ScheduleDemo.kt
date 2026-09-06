package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.context.withSchedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
// end::imports[]

/**
 * SKEEP-005 walk-through: the same attention op on three schedules, bit-identical results, and
 * the trace that says which schedule ran. Every region is included verbatim into
 * `tutorials/schedule-getting-started.adoc`; `SamplesTest` executes it.
 */
object ScheduleDemo {

    class Result(
        val defaultScheduleName: String,
        val sequential: FloatArray,
        val scheduled: FloatArray,
        val regions: List<TraceEvent.ScheduleRegion>,
    )

    // tag::operands[]
    /** Q, K, V for 2 batches × 8 heads: [batch, heads, seq, headDim], built through [ctx]. */
    fun operands(ctx: ExecutionContext, seqQ: Int = 64, seqKV: Int = 256, headDim: Int = 32): Triple<Tensor<FP32, Float>, Tensor<FP32, Float>, Tensor<FP32, Float>> {
        fun tensor(seq: Int, seed: Int): Tensor<FP32, Float> {
            val values = FloatArray(2 * 8 * seq * headDim) { i -> (((i * 31 + seed * 17) % 23) - 11) / 11f }
            return ctx.fromFloatArray(Shape(2, 8, seq, headDim), FP32::class, values)
        }
        return Triple(tensor(seqQ, 1), tensor(seqKV, 2), tensor(seqKV, 3))
    }

    fun attention(ctx: ExecutionContext): FloatArray {
        val (q, k, v) = operands(ctx)
        return ctx.ops.scaledDotProductAttention(q, k, v, mask = null, scale = 0f, causal = true).data.copyToFloatArray()
    }
    // end::operands[]

    fun run(): Result {
        // tag::sequential[]
        val ctx = DirectCpuExecutionContext()                       // JVM: CoroutineSchedule.hardware()
        val defaultName = ctx.schedule.name                          // e.g. "coroutines(12)"
        val sequential = ctx.withSchedule(Schedule.Sequential) { seq ->
            attention(seq)                                           // one task, the caller's thread
        }
        // end::sequential[]

        // tag::scheduled[]
        val sink = RecordingTraceSink()
        val twoWorkers = CoroutineSchedule(parallelism = 2, sink = sink)   // Dispatchers.Default
        val scheduled = ctx.withSchedule(twoWorkers) { par ->
            attention(par)                                           // 16 (batch, head) units on 2 tasks
        }
        // end::scheduled[]

        // tag::trace[]
        val regions = sink.eventsOf<TraceEvent.ScheduleRegion>()
        val identical = sequential.contentEquals(scheduled)
        println("default schedule: $defaultName")
        for (r in regions) println("region: ${r.op} on ${r.schedule} — ${r.elements} units in ${r.tasks} tasks")
        println("outputs identical: $identical")
        // end::trace[]
        return Result(defaultName, sequential, scheduled, regions)
    }
}
