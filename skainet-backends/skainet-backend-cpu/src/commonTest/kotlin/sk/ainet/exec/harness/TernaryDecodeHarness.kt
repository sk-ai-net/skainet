package sk.ainet.exec.harness

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.TernaryKernelPacks
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.memory.ModelScope
import sk.ainet.lang.memory.ProcessMemorySample
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.sample
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.decodeStep
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.DefaultKvCacheStore
import sk.ainet.lang.tensor.storage.KvCacheConfig
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * The M2 shape of the M1 decode harness (#1042): **ternary** weights, an int8 activation adapter,
 * a sliding-window KV ring, and the process-level counters the milestone is judged on.
 *
 * Everything here is the real machinery — `TernaryCodec` weights in a [ModelScope],
 * `I8Absmax.requantize` into the [ForwardScope] every step, dispatch through [KernelDispatch] onto
 * `bitnet_gemv`, a KV ring that wraps — so the assertions are about what SKaiNET actually does,
 * not about a mock. It is still not a model: no tokenizer, no checkpoint, no sampling. The
 * BitNet-2B numbers belong to the decode sample in SKaiNET-transformers; what belongs here is that
 * the memory behaviour holds, on every target, including a 2 GB ARM board.
 */
public class TernaryDecodeHarness(
    public val layers: Int = 2,
    public val hidden: Int = 256,
    public val ctx: Int = 32,
    public val kvHeads: Int = 2,
    public val heads: Int = 4,
) {
    public val sink: RecordingTraceSink = RecordingTraceSink()
    private val model = ModelScope(sink, "m2-harness")

    init {
        require(hidden % 256 == 0) { "the ternary kernel works in TQ2_0 blocks of 256; hidden=$hidden" }
        KernelDispatch.clearForTesting()
        TernaryKernelPacks.install(native = null)     // the portable kernel: what a device without the pack runs
    }

    /** Per-layer ternary weights, TQ2_0, resident in the model scope. */
    private val weights: List<TensorView> = buildList {
        for (l in 0 until layers) {
            add(ternaryWeight(hidden, hidden, TensorId(listOf("model", "layers[$l]", "attn"), "q_proj.weight")))
            add(ternaryWeight(hidden, hidden, TensorId(listOf("model", "layers[$l]", "mlp"), "up_proj.weight")))
        }
    }

    private val kv = DefaultKvCacheStore(
        KvCacheConfig(numLayers = layers, numHeads = kvHeads, headDim = hidden / heads, maxSeqLen = ctx),
        model,
        slidingWindow = true,                          // #1036: the ring, so a long run does not grow
    )

    private val forward = ForwardScope(slabFloats = 4 * hidden + 64, sink = sink, name = "m2-decode")

    /** Bytes of ternary weight resident in the model scope. */
    public val weightBytes: Long = weights.sumOf { it.format.physicalBytes(it.elementCount) ?: 0L }

    private fun ternaryWeight(rows: Int, cols: Int, id: TensorId): TensorView {
        var seed = id.canonical.hashCode()
        val values = FloatArray(rows * cols) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val bytes = TernaryCodec.encode(TensorEncoding.TQ2_0, values)
        val storage = model.adopt(Storage.Heap.wrap(bytes, mutable = false, origin = id, sink = sink))
        return TensorView.packed(
            storage, Shape(rows, cols), TensorEncoding.TQ2_0,
            TernaryBlockDecoder(TensorEncoding.TQ2_0), id = id,
        )
    }

    /** Run [steps] decode steps; returns the process counters before and after the timed region. */
    public fun decode(steps: Int): Pair<ProcessMemorySample, ProcessMemorySample> {
        val activation = FloatArray(hidden) { (it % 13) * 0.0625f }
        val k = FloatArray(kvHeads * (hidden / heads)) { 0.25f }
        // warm up outside the measured window: first-touch faults are not steady-state behaviour
        step(0, activation, k)
        val before = MemoryProbe.sample()
        for (s in 1..steps) step(s, activation, k)
        val after = MemoryProbe.sample()
        return before to after
    }

    private fun step(step: Int, activation: FloatArray, k: FloatArray) {
        sink.decodeStep(step) {
            val slab = forward.allocateFloats(hidden, TensorId(listOf("model"), "hidden", "step=$step"))
            activation.copyInto(slab.floats!!, slab.arrayOffset)
            val dense = TensorView.dense(slab, Shape(1, hidden), FP32, TensorId(listOf("model"), "hidden", "step=$step"))
            for (w in weights) {
                val out = forward.allocateFloats(w.shape[0], TensorId(listOf("model"), "proj", "step=$step"))
                val outView = TensorView.dense(out, Shape(1, w.shape[0]), FP32)
                // the dispatcher requantizes the activation into the forward scope and picks bitnet_gemv
                KernelDispatch.matmul(dense, w, outView, forward, sink)
            }
            for (l in 0 until layers) kv.appendToken(l, k, k)
            forward.reset()
        }
    }

    /** Live bytes per scope as the event stream saw them. */
    public fun liveBytes(): Map<ScopeKind, Long> {
        val live = HashMap<ScopeKind, Long>()
        for (e in sink.events()) when (e) {
            is TraceEvent.Allocation -> live[e.scope] = (live[e.scope] ?: 0L) + e.bytes
            is TraceEvent.Free -> live[e.scope] = ((live[e.scope] ?: 0L) - e.bytes).coerceAtLeast(0L)
            is TraceEvent.ScopeReset -> live[e.scope] = e.liveBytesAfter
            else -> Unit
        }
        return live
    }

    /** Adapter events recorded during step [step]. */
    public fun adaptersInStep(step: Int): List<TraceEvent.AdapterInserted> {
        val out = ArrayList<TraceEvent.AdapterInserted>()
        var current = 0
        for (e in sink.events()) {
            if (e is TraceEvent.PhaseBegin && e.phase == "decode") current = e.step ?: current
            if (e is TraceEvent.AdapterInserted && current == step) out += e
        }
        return out
    }

    public fun close() {
        forward.close()
        model.close()
        KernelDispatch.clearForTesting()
    }
}
