package sk.ainet.exec.harness.android

import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.jni.JniKernelProvider
import sk.ainet.exec.kernel.jni.JniMappedKernelPack
import sk.ainet.io.MappedRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.planInput
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.memory.ModelScope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.sample
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.plan.ActualMemory
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.PlanVsActual
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.DefaultKvCacheStore
import sk.ainet.lang.tensor.storage.KvCacheConfig
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32

/**
 * #1130 (M2-A5): a Q4_K_M model loading and decoding **on this device, under ART's real heap
 * cap**, with RSS and major-fault counts recorded across the run and the [MemoryPlans] plan
 * compared against what was actually allocated.
 *
 * This is a **measurement harness, not an acceptance test** (the issue is explicit: numbers are
 * recorded, not turned into CI assertions — a shared runner is not the instrument for this
 * claim). It therefore never calls [PlanVsActual.check]; it renders the comparison into the
 * report and skips cleanly when no model file is present, so `connectedAndroidTest` on a
 * model-less runner stays green.
 *
 * ## Running it (device with the model)
 *
 * 1. Install the test APK once so its external dir exists, then push the model where the app
 *    can read it (its own external-files dir — `/data/local/tmp` is not readable by an app
 *    process on SELinux-enforcing devices):
 *    ```
 *    adb shell mkdir -p /sdcard/Android/data/sk.ainet.exec.kernel.jni.test/files
 *    adb push model-q4_k_m.gguf /sdcard/Android/data/sk.ainet.exec.kernel.jni.test/files/skainet-m2a5.gguf
 *    ```
 * 2. Run just this class (knobs: `ctx`, `steps`, `modelPath` — all optional):
 *    ```
 *    ./gradlew :skainet-backends:skainet-backend-jni-cpu:connectedDebugAndroidTest \
 *      -Pandroid.testInstrumentationRunnerArguments.class=sk.ainet.exec.harness.android.M2A5DeviceMeasurement \
 *      -Pandroid.testInstrumentationRunnerArguments.ctx=512 \
 *      -Pandroid.testInstrumentationRunnerArguments.steps=16
 *    ```
 * 3. Pull the report and paste it into #1130:
 *    ```
 *    adb pull /sdcard/Android/data/sk.ainet.exec.kernel.jni.test/files/m2a5-report.md
 *    ```
 *    The same report also streams to logcat under the `M2A5` tag.
 *
 * ## What the decode loop is
 *
 * The engine owns no tokenizer or sampler (those live in SKaiNET-transformers), so "decodes"
 * here is the [sk.ainet.exec.harness] DecodeHarness shape over the **real loaded weights**: per
 * step, every layer's attention and FFN projections run as packed matmuls through
 * [KernelDispatch] out of a recycled [ForwardScope], and one token's K/V goes into a
 * model-scoped ring — the exact memory traffic of a decode step, which is what M2-A5 measures.
 */
@RunWith(AndroidJUnit4::class)
class M2A5DeviceMeasurement {

    private val args = InstrumentationRegistry.getArguments()
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private fun arg(name: String, default: Int): Int = args.getString(name)?.toIntOrNull() ?: default

    @Test
    fun measure_q4km_under_real_heap_cap() {
        val defaultPath = File(context.getExternalFilesDir(null), "skainet-m2a5.gguf").path
        val modelPath = args.getString("modelPath") ?: defaultPath
        val modelFile = File(modelPath)
        assumeTrue(
            "No model at $modelPath — push a Q4_K_M GGUF there (see the class KDoc) to record " +
                "the #1130 measurement; skipping.",
            modelFile.canRead(),
        )
        val ctxLen = arg("ctx", 512)
        val prepack = args.getString("prepack")?.toBoolean() ?: false
        val steps = arg("steps", 16)
        val warmup = arg("warmup", 4)
        // `residency=heap` (default mapped) restores heap staging — the #1193 A/B lever for
        // models that fit the cap. The plan below prices the same form the load uses (#1190).
        val residency = if (args.getString("residency") == "heap") WeightResidency.HEAP else WeightResidency.MAPPED
        val loadForm = WeightForm(shape = WeightShapeOrientation.OUT_IN, residency = residency)

        val report = StringBuilder()
        fun line(s: String = "") { report.append(s).append('\n') }
        fun mb(b: Long?) = if (b == null) "—" else MemoryPlans.formatBytes(b)

        val heapCap = Runtime.getRuntime().maxMemory()
        line("# M2-A5 measurement (#1130)")
        line()
        line("- device: ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT}), ABI ${Build.SUPPORTED_ABIS.firstOrNull()}")
        line("- ART heap cap (Runtime.maxMemory): ${mb(heapCap)}")
        line("- model: ${modelFile.name}, ${mb(modelFile.length())} on disk")
        line("- ctx=$ctxLen, decode steps=$steps (warm-up $warmup), prepack=$prepack, residency=${residency.name.lowercase()}")
        line()

        // ---- plan, from the header only --------------------------------------------------
        val (plan, geometry) = MappedRandomAccessSource.open(modelPath).let { src ->
            StreamingGGUFReader.open(src).use { reader ->
                val input = reader.planInput(ctx = ctxLen, formFor = { loadForm })
                MemoryPlans.plan(input, Budget.of(heapCap)) to input.geometry
            }
        }
        line("## Plan"); line(); line("```"); line(plan.render().trimEnd()); line("```"); line()

        // ---- load, mapped residency, everything traced -----------------------------------
        KernelDispatch.clearForTesting()
        runCatching { KernelPacks.install(JniKernelProvider); KernelPacks.installPacked(JniKernelProvider) }
            .onFailure { line("_note: JNI kernel pack unavailable (${it.message}); reference kernels serve — memory numbers stay valid, timings do not._") }
        // #1189: row-major direct-buffer kernels — these serve the mapped packed weights below
        // with zero copies and no prepack.
        runCatching { JniMappedKernelPack.install() }
            .onFailure { line("_note: mapped kernel pack unavailable (${it.message}); mapped packed weights fall back to the decoding reference._") }

        val sink = RecordingTraceSink()
        val ctx = DirectCpuExecutionContext()
        val sBefore = MemoryProbe.sample()
        val tLoad0 = System.nanoTime()
        val tensors = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { MappedRandomAccessSource.open(modelPath) },
                weightForm = loadForm,
                traceSink = sink,
            ).load<FP32, Float>(ctx, FP32::class) { name, t -> tensors[name] = t }
        }
        val loadMs = (System.nanoTime() - tLoad0) / 1_000_000
        val sLoaded = MemoryProbe.sample()

        // Account the materialized weights as model-scope allocations so plan-vs-actual sees
        // them; the loader's own storages are not all sink-wired. #1189 note: physicalBytes
        // (not packedData.size) — a mapped packed tensor keeps its bytes off-heap and has no
        // heap ByteArray at all; those bytes are counted separately as mapped.
        var weightHeapBytes = 0L
        var weightMappedBytes = 0L
        var mappedPackedCount = 0
        var syntheticStorageId = -1_000_000L
        for ((name, t) in tensors) {
            val d = t.data
            val bytes = (d as? PackedBlockStorage)?.physicalBytes ?: (t.shape.volume.toLong() * 4L)
            if (d is sk.ainet.lang.tensor.data.BufferPackedTensorData) {
                weightMappedBytes += bytes; mappedPackedCount += 1
            } else {
                weightHeapBytes += bytes
            }
            syntheticStorageId -= 1
            sink.emit(TraceEvent.Allocation(syntheticStorageId, ScopeKind.MODEL, bytes, null, "m2a5:$name"))
        }
        line("## Load")
        line()
        line("- wall time: ${loadMs} ms for ${tensors.size} tensors")
        line("- heap bytes (dense tensors + heap packed payloads): ${mb(weightHeapBytes)}")
        line("- mapped packed payloads (#1189, off-heap): ${mb(weightMappedBytes)} in $mappedPackedCount tensors")
        line("- RSS before → after load: ${mb(sBefore.rssBytes)} → ${mb(sLoaded.rssBytes)} (Δ ${mb((sLoaded.rssBytes ?: 0L) - (sBefore.rssBytes ?: 0L))})")
        line("- major faults during load: ${sLoaded.majorFaultsSince(sBefore) ?: "—"}")
        line()

        // ---- decode loop over the real weights -------------------------------------------
        val g = geometry
        assumeTrue("GGUF header lacks architecture geometry; cannot shape the decode loop", g != null)
        g!!
        val model = ModelScope(sink, "m2a5")
        val kv = DefaultKvCacheStore(
            KvCacheConfig(numLayers = g.layers, numHeads = g.kvHeads, headDim = g.headDim, maxSeqLen = ctxLen),
            model,
        )
        val slabFloats = (4 * g.embeddingLength + 3 * g.feedForwardLength + g.heads * ctxLen) + g.vocabSize
        val forward = ForwardScope(slabFloats = slabFloats, sink = sink, name = "m2a5-decode")

        // With prepack=true, each used weight is permuted ONCE into the feed order the JNI
        // packed kernels read (installPacked keys on BLOCKED_INPUT_MAJOR); without it the
        // decoding reference serves — memory numbers identical, timings reference-grade. The
        // prepack copies are extra resident bytes and show in the trace: that cost is itself a
        // finding this harness records.
        fun weightView(name: String): TensorView? {
            val v = (tensors[name]?.data as? PackedBlockStorage)?.packedView ?: return null
            return if (!prepack) v else runCatching {
                v.prepack(sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR, sink = sink)
            }.getOrDefault(v)
        }
        val layerViews = (0 until g.layers).map { l ->
            listOfNotNull(
                weightView("blk.$l.attn_q.weight"), weightView("blk.$l.attn_k.weight"),
                weightView("blk.$l.attn_v.weight"), weightView("blk.$l.attn_output.weight"),
                weightView("blk.$l.ffn_gate.weight"), weightView("blk.$l.ffn_up.weight"),
            ) to weightView("blk.$l.ffn_down.weight")
        }

        data class StepSample(val step: Int, val ms: Long, val rss: Long?, val majDelta: Long?)
        val samples = ArrayList<StepSample>()
        var prev = sLoaded
        val kRow = FloatArray(g.kvHeads * g.headDim) { 0.25f }
        for (step in 1..steps) {
            val t0 = System.nanoTime()
            val act = forward.allocateFloats(g.embeddingLength, null)
            act.floats!!.fill(0.1f, act.arrayOffset, act.arrayOffset + g.embeddingLength)
            val actView = TensorView.dense(act, Shape(1, g.embeddingLength), FP32)
            for ((embWeights, down) in layerViews) {
                var ffnAct: TensorView? = null
                for (w in embWeights) {
                    val out = forward.allocateFloats(w.shape[0], null)
                    val outView = TensorView.dense(out, Shape(1, w.shape[0]), FP32)
                    KernelDispatch.matmul(actView, w, outView, forward, sink)
                    if (w.shape[0] == g.feedForwardLength) ffnAct = outView
                }
                if (down != null && ffnAct != null) {
                    val out = forward.allocateFloats(down.shape[0], null)
                    KernelDispatch.matmul(ffnAct, down, TensorView.dense(out, Shape(1, down.shape[0]), FP32), forward, sink)
                }
            }
            if (kv.currentSeqLen < ctxLen) for (l in 0 until g.layers) kv.appendToken(l, kRow, kRow)
            forward.reset()
            val ms = (System.nanoTime() - t0) / 1_000_000
            val s = MemoryProbe.sample()
            samples += StepSample(step, ms, s.rssBytes, s.majorFaultsSince(prev))
            prev = s
        }

        // ---- plan vs actual (rendered, deliberately not check()ed) -----------------------
        val actual = ActualMemory.from(sink)
        val pva = PlanVsActual(plan, actual)
        line("## Decode — per-step RSS and major faults")
        line()
        line("| step | ms | RSS | Δ major faults |")
        line("|---|---|---|---|")
        for (s in samples) line("| ${s.step} | ${s.ms} | ${mb(s.rss)} | ${s.majDelta ?: "—"} |")
        val steady = samples.drop(warmup)
        if (steady.isNotEmpty()) {
            line()
            line("- steady-state (after $warmup warm-up steps): ${steady.map { it.ms }.average().toInt()} ms/step, " +
                "major faults total ${steady.mapNotNull { it.majDelta }.sum()}")
            // #1193: reference fallbacks are trace events now — a nonzero count here is the
            // 48 s/step trap announcing itself instead of hiding in the timings.
            val fallbacks = sink.events().filterIsInstance<TraceEvent.KernelRun>()
                .filter { "reference-fallback" in it.kernel }
            line("- packed-kernel reference fallbacks: ${fallbacks.size}" +
                if (fallbacks.isEmpty()) "" else " — e.g. ${fallbacks.first().kernel}")
            line("- forward-scope allocations after warm-up should be 0; " +
                "whole-run forward allocations: ${actual.allocationsByScope[ScopeKind.FORWARD] ?: 0}")
        }
        line()
        line("## Plan vs actual"); line(); line("```"); line(pva.render().trimEnd()); line("```")
        line()
        line("_within ${(pva.tolerance * 100).toInt()} % tolerance: ${pva.withinTolerance} " +
            "(recorded as a measurement, per #1130 — not asserted)_")

        forward.close(); model.close()

        // ---- emit ------------------------------------------------------------------------
        val text = report.toString()
        val out = File(context.getExternalFilesDir(null), "m2a5-report.md")
        runCatching { out.writeText(text) }
        for (chunk in text.chunked(3000)) Log.i("M2A5", chunk)
        println(text)
    }
}
