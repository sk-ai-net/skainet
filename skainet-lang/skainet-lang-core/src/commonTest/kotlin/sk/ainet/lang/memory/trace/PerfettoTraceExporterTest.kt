package sk.ainet.lang.memory.trace

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** SKEEP-003 §4.9 / PRD M1-A7: one track per scope, kernel spans labelled by TensorId, a live-bytes counter. */
class PerfettoTraceExporterTest {

    private fun decodeStepEvents(): List<TraceEvent> {
        val w = TensorId.parse("model.layers[0].attn.q_proj.weight")
        val act = TensorId.parse("model.layers[0].attn.q#step=1")
        return listOf(
            TraceEvent.Plan("llama-1b", 2048, 800L shl 20, 64L shl 20, 47L shl 20, 64L shl 20, 1300L shl 20, true, timeNanos = 0),
            TraceEvent.Allocation(1, ScopeKind.MODEL, 800L shl 20, w, site = "model.gguf", timeNanos = 1_000),
            TraceEvent.PhaseBegin("decode", 1, mapOf("tokens" to "1"), timeNanos = 2_000),
            TraceEvent.Allocation(2, ScopeKind.FORWARD, 8192, act, timeNanos = 2_500),
            TraceEvent.KernelRun("matmul", "scalar-q8_0", listOf(act, w), act, 4096, 64, durationNanos = 500_000, timeNanos = 3_000_000),
            TraceEvent.AdapterInserted("dequantize", Format(FP32, TensorEncoding.Q6_K), Format.dense(FP32), 96L shl 20, w, timeNanos = 3_100_000),
            TraceEvent.ScopeReset(ScopeKind.FORWARD, 8192, 0, timeNanos = 3_200_000),
            TraceEvent.PhaseEnd("decode", 1, durationNanos = 3_198_000, timeNanos = 3_200_000),
            TraceEvent.Counter("rss", 900L shl 20, timeNanos = 3_300_000),
        )
    }

    @Test
    fun rendersAChromeTraceWithTracksSlicesCountersAndArgs() {
        val json = PerfettoTraceExporter.export(decodeStepEvents(), processName = "skainet-decode")
        assertTrue(json.startsWith("{\"traceEvents\":["), "must be a Chrome trace document")
        assertTrue(json.trimEnd().endsWith("\"displayTimeUnit\":\"ms\"}"))
        // process / thread naming: one track per scope
        assertTrue(json.contains("\"name\":\"process_name\"") && json.contains("skainet-decode"))
        assertTrue(json.contains("\"name\":\"thread_name\"") && json.contains("model scope") && json.contains("forward scope"))
        // phases as B/E slices, kernels as complete slices with a duration
        assertTrue(json.contains("\"ph\":\"B\",\"name\":\"decode#1\""), json.take(400))
        assertTrue(json.contains("\"ph\":\"E\",\"name\":\"decode#1\""))
        assertTrue(json.contains("\"ph\":\"X\",\"name\":\"matmul\""))
        assertTrue(json.contains("\"dur\":500.000"), "kernel duration in microseconds")
        // labelled by TensorId
        assertTrue(json.contains("model.layers[0].attn.q_proj.weight"))
        assertTrue(json.contains("\"kernel\":\"scalar-q8_0\""))
        // adapters visible with their byte cost
        assertTrue(json.contains("adapter:dequantize") && json.contains("\"bytes\":\"100663296\""))
        // live-bytes counters per scope
        assertTrue(json.contains("\"ph\":\"C\",\"name\":\"live bytes\""))
        assertTrue(json.contains("\"forward\":8192") && json.contains("\"forward\":0"), "the reset must drop the counter back to zero")
        assertTrue(json.contains("\"model\":838860800"))
        // plan and platform counters
        assertTrue(json.contains("\"name\":\"plan\"") && json.contains("\"fits\":\"true\""))
        assertTrue(json.contains("\"name\":\"rss\""))
    }

    @Test
    fun exportsWhatARecordingSinkKept() {
        val sink = RecordingTraceSink()
        sink.phase("prefill", 0) { sink.emit(TraceEvent.Counter("tokens", 64, "count")) }
        val json = PerfettoTraceExporter.export(sink)
        assertTrue(json.contains("prefill#0"))
        assertTrue(json.contains("\"name\":\"tokens\""))
        assertEquals(3, sink.events().size)
    }

    @Test
    fun escapesStringsAndFormatsTimestamps() {
        val json = PerfettoTraceExporter.export(listOf(TraceEvent.PhaseBegin("we\"ird\n", null, mapOf("k" to "a\\b"), timeNanos = 1_500)))
        assertTrue(json.contains("we\\\"ird\\n"), json)
        assertTrue(json.contains("\"a\\\\b\""))
        assertTrue(json.contains("\"ts\":1.500"), "nanoseconds render as microseconds")
    }
}
