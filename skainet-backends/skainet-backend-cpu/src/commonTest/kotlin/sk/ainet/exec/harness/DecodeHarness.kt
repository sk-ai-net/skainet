package sk.ainet.exec.harness

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.ModelScope
import sk.ainet.lang.memory.PackedBlockDecoder
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.plan.KvCacheMode
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.ModelGeometry
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.PlanTensor
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.decodeStep
import sk.ainet.lang.memory.trace.module
import sk.ainet.lang.memory.trace.phase
import sk.ainet.lang.memory.trace.prefill
import sk.ainet.lang.memory.trace.sample
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.DefaultKvCacheStore
import sk.ainet.lang.tensor.storage.KvCacheConfig
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A synthetic decode loop over the *real* memory machinery — SKEEP-003 M1's acceptance harness
 * (#1032, option (c)).
 *
 * It is deliberately not a model: no tokenizer, no sampling, no checkpoint. It is a Llama-shaped
 * stack of packed matmuls whose weights live in a [ModelScope] (mapped-equivalent, packed Q8_0),
 * whose activations come from a recycled [ForwardScope], whose KV ring is preallocated in the model
 * scope, and whose dispatch goes through [KernelDispatch] — so the criteria that are about *memory
 * behaviour* can be asserted on every commit in the repository where that behaviour lives:
 *
 * - **M1-A1** flat memory: live bytes at the last step equal live bytes after warm-up;
 * - **M1-A3** zero forward-scope allocations per step after warm-up;
 * - **M1-A8** the plan matches what the run allocated;
 * - **M1-A7** the trace has one track per scope and a live-bytes counter that returns to zero.
 *
 * The real end-to-end numbers (tok/s, TTFT, peak load RSS, effective bandwidth) belong to the
 * `skainet-decode` sample in SKaiNET-transformers, which owns the model.
 */
public class DecodeHarness(
    // Deliberately tiny: the reference kernel decodes every element, and these tests run in a
    // browser under Karma's 2 s per-test budget as well as on the JVM. The assertions are about
    // *memory behaviour*, which does not need big shapes — the real numbers come from
    // skainet-decode in SKaiNET-transformers.
    public val layers: Int = 2,
    public val hidden: Int = 32,
    public val ffn: Int = 64,
    public val heads: Int = 4,
    public val kvHeads: Int = 2,
    public val ctx: Int = 32,
    public val vocab: Int = 64,
) {
    public val sink: RecordingTraceSink = RecordingTraceSink()
    private val model = ModelScope(sink, "harness")
    private val blockSize = 32

    /** Per-layer weights, packed Q8_0, living in the model scope (the mapped-weight shape). */
    private val weights: List<TensorView> = buildList {
        for (l in 0 until layers) {
            add(packedWeight(hidden, hidden, TensorId(listOf("model", "layers[$l]", "attn"), "q_proj.weight")))
            add(packedWeight(ffn, hidden, TensorId(listOf("model", "layers[$l]", "mlp"), "up_proj.weight")))
        }
    }

    private val kv = DefaultKvCacheStore(
        KvCacheConfig(numLayers = layers, numHeads = kvHeads, headDim = hidden / heads, maxSeqLen = ctx),
        model,
    )

    /** The forward slab: activations of one step, sized from the plan. */
    private val forward = ForwardScope(slabFloats = forwardFloats(), sink = sink, name = "decode")

    private fun forwardFloats(): Int = (4 * hidden + 3 * ffn + heads * ctx) + vocab

    private fun packedWeight(rows: Int, cols: Int, id: TensorId): TensorView {
        require(cols % blockSize == 0)
        val blocks = rows * (cols / blockSize)
        val bytes = ByteArray(blocks * 34)
        var seed = id.canonical.hashCode()
        for (b in 0 until blocks) {
            val off = b * 34
            // a sane FP16 scale, then deterministic codes
            bytes[off] = 0x00; bytes[off + 1] = 0x38          // half(0.5)
            for (i in 0 until 32) { seed = seed * 1103515245 + 12345; bytes[off + 2 + i] = (seed ushr 16).toByte() }
        }
        val storage = model.adopt(Storage.Heap.wrap(bytes, mutable = false, origin = id, sink = sink))
        val data = Q8_0BlockTensorData(Shape(rows, cols), bytes)
        return TensorView.packed(storage, Shape(rows, cols), TensorEncoding.Q8_0, PackedBlockDecoder(data), id = id)
    }

    /** The memory plan this harness's shapes predict. */
    public fun plan(): sk.ainet.lang.memory.plan.MemoryPlan {
        val f = Format(FP32, TensorEncoding.Q8_0)
        val tensors = weights.map { w ->
            PlanTensor(w.id!!.canonical, w.id, f, w.elementCount, f.physicalBytes(w.elementCount)!!)
        }
        val geometry = ModelGeometry(layers, heads, kvHeads, hidden / heads, hidden / heads, hidden, ffn, vocab)
        return MemoryPlans.plan(PlanInput("harness", "llama", tensors, geometry, ctx, prefillChunk = 1, kvMode = KvCacheMode.FP32))
    }

    /**
     * The prompt pass: [tokens] positions through the same stack, inside a `prefill` span so
     * [metrics] can price it (#1035). Deliberately the same work as a decode step — the harness is
     * about memory behaviour, not about being a fast prefill.
     */
    public fun prefill(tokens: Int) {
        sink.prefill(tokens) {
            repeat(tokens) { runStack(step = 0) }
        }
    }

    /** Run [steps] decode steps; each allocates activations, runs the stack and resets the scope. */
    public fun decode(steps: Int) {
        for (step in 1..steps) {
            sink.decodeStep(step) {
                runStack(step)
                // one token into the KV ring (all layers), as a decode step does
                val k = FloatArray(kvHeads * (hidden / heads)) { 0.5f }
                if (kv.currentSeqLen < ctx) for (l in 0 until layers) kv.appendToken(l, k, k)
                forward.reset()
            }
            sink.sample(step) { /* argmax over a synthetic logit row: nothing to allocate */ }
        }
    }

    /** One pass over the weight stack, each weight timed as its own module span. */
    private fun runStack(step: Int) {
        val x = FloatArray(hidden) { (it % 7) * 0.125f }
        val act = forward.allocateFloats(hidden, TensorId(listOf("model"), "hidden", "step=$step"))
        x.copyInto(act.floats!!, act.arrayOffset)
        val actView = TensorView.dense(act, Shape(1, hidden), FP32, TensorId(listOf("model"), "hidden", "step=$step"))
        for (w in weights) {
            if (w.shape[1] != hidden) continue
            sink.module(w.id!!, step) {
                val out = forward.allocateFloats(w.shape[0], TensorId(listOf("model"), "proj", "step=$step"))
                val outView = TensorView.dense(out, Shape(1, w.shape[0]), FP32)
                KernelDispatch.matmul(actView, w, outView, forward, sink)
            }
        }
    }

    /** The generation metrics this run produced (#1035); [peakBytesPerSecond] enables utilization. */
    public fun metrics(peakBytesPerSecond: Long? = null): sk.ainet.lang.memory.trace.GenerationMetrics =
        sk.ainet.lang.memory.trace.GenerationMetrics.from(sink, peakBytesPerSecond)

    /** Live bytes per scope as the event stream saw them after the last step. */
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

    /** Allocation events recorded in [scope] between two step numbers (inclusive of the phases). */
    public fun allocationsBetweenSteps(scope: ScopeKind, fromStep: Int, toStep: Int): Int {
        var step = 0
        var count = 0
        for (e in sink.events()) {
            if (e is TraceEvent.PhaseBegin && e.phase == "decode") step = e.step ?: step
            if (e is TraceEvent.Allocation && e.scope == scope && step in fromStep..toStep) count++
        }
        return count
    }

    public fun close() { forward.close(); model.close() }
}
