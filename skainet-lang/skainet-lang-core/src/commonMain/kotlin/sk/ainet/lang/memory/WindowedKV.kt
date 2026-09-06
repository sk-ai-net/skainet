package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId

/**
 * An attention window over a KV ring, as the **one or two pieces it physically is** (SKEEP-003
 * §4.6, decision #4; M2-F5).
 *
 * A ring that has wrapped holds its newest `n` positions in two runs — `[from, capacity)` and
 * `[0, wrapEnd)` — and the choice has always been to copy them together before attention or to
 * grow the buffer forever. This is the third option: hand the kernel the pair. Both halves are
 * ordinary [TensorView]s over the same [Storage], so nothing is copied and nothing is special-cased;
 * a kernel that cannot iterate a pair calls [gather], which makes the copy *visible* as one
 * adapter in the trace instead of an invisible per-token allocation.
 *
 * Position order is `head` then `tail`: oldest to newest.
 *
 * @property head the first (older) run — the only one when the window does not wrap
 * @property tail the second (newer) run, present only after the ring wraps
 */
public class WindowedKV(
    public val head: TensorView,
    public val tail: TensorView? = null,
) {
    init {
        val t = tail
        if (t != null) {
            require(t.shape.rank == head.shape.rank) { "both halves must have the same rank" }
            require(t.format == head.format) { "both halves must have the same format" }
            for (axis in 0 until head.shape.rank) {
                if (axis == POSITION_AXIS) continue
                require(t.shape[axis] == head.shape[axis]) { "halves differ on axis $axis: ${head.shape} vs ${t.shape}" }
            }
        }
    }

    /** The halves in position order, oldest first. */
    public val parts: List<TensorView> get() = if (tail == null) listOf(head) else listOf(head, tail)

    /** Positions in the window, across both halves. */
    public val length: Int get() = head.shape[POSITION_AXIS] + (tail?.shape?.get(POSITION_AXIS) ?: 0)

    /** True when the ring wrapped inside this window — the case the pair exists for. */
    public val wrapped: Boolean get() = tail != null

    /** `[heads, dim]` — the shape of the window with its position axis removed. */
    public val heads: Int get() = head.shape[0]
    public val headDim: Int get() = head.shape[head.shape.rank - 1]

    /**
     * The window as one contiguous view in [scope] — the gather adapter (§5.1).
     *
     * For a kernel that cannot iterate the pair. The copy is real, so it is *traced*: one
     * [TraceEvent.AdapterInserted] per call, which is what makes "the zero-copy path allocates
     * nothing per token" an assertion rather than a hope.
     */
    public fun gather(scope: Scope, sink: TraceSink = NoopTraceSink, id: TensorId? = null): TensorView {
        val total = length
        val shape = Shape(heads, total, headDim)
        val storage = scope.allocateFloats(heads * total * headDim, id)
        val out = storage.floats!!
        val base = storage.arrayOffset
        var written = 0
        for (part in parts) {
            val positions = part.shape[POSITION_AXIS]
            for (h in 0 until heads) {
                for (p in 0 until positions) {
                    val dst = base + (h.toLong() * total + written + p).toInt() * headDim
                    for (d in 0 until headDim) out[dst + d] = part.get(h, p, d)
                }
            }
            written += positions
        }
        if (sink.isEnabled) {
            sink.emit(
                TraceEvent.AdapterInserted(
                    kind = "gather-kv-window",
                    from = head.format,
                    to = Format.dense(head.format.dtype),
                    bytes = heads.toLong() * total * headDim * head.format.dtype.sizeInBytes,
                    target = id ?: head.id,
                    scope = scope.kind,
                ),
            )
        }
        return TensorView.dense(storage, shape, head.format.dtype, id ?: head.id)
    }

    /** The decoded value at (`head`, absolute window position, `dim`), crossing the halves. */
    public fun get(head: Int, position: Int, dim: Int): Float {
        require(position in 0 until length) { "position $position outside the window (length $length)" }
        val first = this.head.shape[POSITION_AXIS]
        return if (position < first) this.head.get(head, position, dim)
        else tail!!.get(head, position - first, dim)
    }

    override fun toString(): String =
        "WindowedKV(${heads}h × $length × $headDim, ${if (wrapped) "wrapped: ${head.shape[POSITION_AXIS]}+${tail!!.shape[POSITION_AXIS]}" else "contiguous"})"

    public companion object {
        /** KV windows are `[heads, positions, headDim]`. */
        public const val POSITION_AXIS: Int = 1
    }
}
