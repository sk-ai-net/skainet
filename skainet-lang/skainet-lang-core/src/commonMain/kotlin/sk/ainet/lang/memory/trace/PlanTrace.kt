package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.plan.MemoryPlan

/** Emit this plan as a [TraceEvent.Plan] so the plan-vs-actual check (#1030) can find it in the stream. */
public fun MemoryPlan.emit(sink: TraceSink) {
    if (!sink.isEnabled) return
    sink.emit(
        TraceEvent.Plan(
            model = input.modelName, ctx = input.ctx,
            weightsBytes = weightsBytes, kvBytes = kvBytes, forwardBytes = forwardBytes, headroomBytes = headroomBytes,
            budgetBytes = budget?.bytes, fits = fits,
        ),
    )
}
