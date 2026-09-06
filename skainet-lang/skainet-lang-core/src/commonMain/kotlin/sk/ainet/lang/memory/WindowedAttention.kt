package sk.ainet.lang.memory

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Attention over a [WindowedKV] — the reference kernel that **iterates the pair** instead of
 * requiring one contiguous window (SKEEP-003 §4.6, M2-F5/M2-A4).
 *
 * The softmax is computed online (running maximum and running denominator, the flash-attention
 * recurrence), which is what makes the pair workable: positions are consumed in order, one run
 * after the other, and nothing has to exist as a single array. It also means the kernel allocates
 * **nothing** per token — the running accumulator is the caller's output row.
 *
 * A kernel that cannot do this calls [WindowedKV.gather] instead and gets one contiguous view plus
 * one traced adapter; both paths are asserted to agree.
 */
public object WindowedAttention {

    /**
     * One decode step: `softmax(q·kᵀ · scale) @ v` for every head, over the window's positions in
     * order.
     *
     * @param query `[heads * headDim]` — the current token's query rows
     * @param keys the key window; `[heads, positions, headDim]` across its runs
     * @param values the value window, the same shape as [keys]
     * @param out `[heads * headDim]`, overwritten with the attention output
     * @param scale multiplier on the scores; `0` means the usual `1/sqrt(headDim)`
     */
    public fun decodeStep(
        query: FloatArray,
        keys: WindowedKV,
        values: WindowedKV,
        out: FloatArray,
        scale: Float = 0f,
    ) {
        val heads = keys.heads
        val headDim = keys.headDim
        require(keys.length == values.length) { "key and value windows differ: ${keys.length} vs ${values.length}" }
        require(values.heads == heads && values.headDim == headDim) { "key and value windows must have the same geometry" }
        require(query.size == heads * headDim) { "query must be [heads * headDim] = ${heads * headDim}, was ${query.size}" }
        require(out.size == heads * headDim) { "out must be [heads * headDim] = ${heads * headDim}, was ${out.size}" }
        require(keys.length > 0) { "cannot attend over an empty window" }
        val s = if (scale != 0f) scale else 1f / sqrt(headDim.toFloat())

        for (h in 0 until heads) {
            val base = h * headDim
            for (d in 0 until headDim) out[base + d] = 0f
            var runningMax = Float.NEGATIVE_INFINITY
            var denominator = 0f
            var position = 0
            // Runs in order, oldest first: head, then tail when the ring wrapped.
            for (partIndex in keys.parts.indices) {
                val k = keys.parts[partIndex]
                val v = values.parts[partIndex]
                require(k.shape[WindowedKV.POSITION_AXIS] == v.shape[WindowedKV.POSITION_AXIS]) {
                    "key and value runs must line up (run $partIndex)"
                }
                for (p in 0 until k.shape[WindowedKV.POSITION_AXIS]) {
                    var score = 0f
                    for (d in 0 until headDim) score += query[base + d] * k.get(h, p, d)
                    score *= s

                    val newMax = max(runningMax, score)
                    val correction = if (runningMax == Float.NEGATIVE_INFINITY) 0f else exp(runningMax - newMax)
                    val weight = exp(score - newMax)
                    denominator = denominator * correction + weight
                    for (d in 0 until headDim) {
                        out[base + d] = out[base + d] * correction + weight * v.get(h, p, d)
                    }
                    runningMax = newMax
                    position++
                }
            }
            for (d in 0 until headDim) out[base + d] = out[base + d] / denominator
        }
    }

    /**
     * [decodeStep] through the gather adapter: the window is copied into [scope] as one contiguous
     * view first, for kernels that cannot iterate a pair. Same numbers, one allocation and one
     * traced adapter per call.
     */
    public fun decodeStepGathered(
        query: FloatArray,
        keys: WindowedKV,
        values: WindowedKV,
        out: FloatArray,
        scope: Scope,
        sink: sk.ainet.lang.memory.trace.TraceSink = sk.ainet.lang.memory.trace.NoopTraceSink,
        scale: Float = 0f,
    ) {
        decodeStep(query, WindowedKV(keys.gather(scope, sink)), WindowedKV(values.gather(scope, sink)), out, scale)
    }
}
