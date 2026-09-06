package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1117: a weight that is re-encoded on the way in says so; one that is not stays silent.
 *
 * The question this makes answerable is "why is this model bigger than the file". Before, the
 * answer lived in whoever remembered which policy was set.
 */
class WeightFormTraceTest {

    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("blk.0.attn_q.weight", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("blk.0.ffn_up.weight", GGMLQuantizationType.Q8_0, elements = 1024),
        SyntheticGguf.tensor("token_embd.weight", GGMLQuantizationType.F32, elements = 512),
    )

    private fun load(f: File, form: WeightForm?, sink: RecordingTraceSink) {
        val ctx = DefaultDataExecutionContext()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
                traceSink = sink,
            ).load<FP32, Float>(ctx, FP32::class) { _, _ -> }
        }
    }

    private fun conversions(sink: RecordingTraceSink): List<TraceEvent.AdapterInserted> =
        sink.events().filterIsInstance<TraceEvent.AdapterInserted>()

    @Test
    fun `a weight held as stored converts nothing and says nothing`() {
        val f = file()
        try {
            val sink = RecordingTraceSink()
            load(f, WeightForm.AS_STORED_ON_HEAP, sink)
            assertTrue(
                conversions(sink).isEmpty(),
                "the common case must stay silent, got ${conversions(sink).map { it.kind }}",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `a dequantized weight reports the tensor and both sizes`() {
        val f = file()
        try {
            val sink = RecordingTraceSink()
            load(f, WeightForm(encoding = EncodingRequest.DequantizeTo(FP32)), sink)

            val events = conversions(sink)
            assertEquals(2, events.size, "one per quantized weight; F32 needs no conversion: ${events.map { it.kind }}")

            val q4k = events.single { it.from.encoding == TensorEncoding.Q4_K }
            assertEquals("dequantize-on-load", q4k.kind)
            assertEquals("blk.0.attn_q.weight", q4k.target?.canonical, "the event names the tensor")
            assertEquals(1024L * 4, q4k.bytes, "afterwards: dense FP32")
            assertTrue(q4k.bytesBefore < q4k.bytes, "before: the packed bytes, ${q4k.bytesBefore}")
            assertEquals(q4k.bytes - q4k.bytesBefore, q4k.bytesDelta, "the delta is what it cost")
            assertTrue(q4k.bytesDelta > 0, "a dequantization only ever adds")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the widening nobody asked for is the one most worth seeing`() {
        // Ternary tensors widen to FP32 whatever the policy, because packed ternary storage does
        // not exist yet (#1033). No flag reveals that; the trace does.
        val f = SyntheticGguf.write(
            SyntheticGguf.tensor("blk.0.ffn_down.weight", GGMLQuantizationType.TQ2_0, elements = 512),
        )
        try {
            val sink = RecordingTraceSink()
            load(f, WeightForm.AS_STORED_ON_HEAP, sink)

            val event = conversions(sink).single()
            assertEquals("widen-ternary-no-packed-storage", event.kind)
            assertEquals(512L * 4, event.bytes)
            assertTrue(
                event.bytes > event.bytesBefore * 5,
                "a ~2-bit weight as FP32 is a large multiple: ${event.bytesBefore} → ${event.bytes}",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `a narrow float widened at load is reported too`() {
        val f = SyntheticGguf.write(SyntheticGguf.tensor("w", GGMLQuantizationType.F16, elements = 256))
        try {
            val sink = RecordingTraceSink()
            load(f, WeightForm.AS_STORED_ON_HEAP, sink)

            val event = conversions(sink).single()
            assertEquals("widen-f16", event.kind)
            assertEquals(256L * 2, event.bytesBefore)
            assertEquals(256L * 4, event.bytes, "F16 to FP32 doubles it")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the default loader records nothing at all`() {
        // NoopTraceSink is the default, so a caller who never asked for a trace pays for none.
        val f = file()
        try {
            val ctx = DefaultDataExecutionContext()
            runBlocking {
                StreamingGgufParametersLoader(sourceProvider = { JvmRandomAccessSource.open(f) })
                    .load<FP32, Float>(ctx, FP32::class) { _, _ -> }
            }
        } finally {
            f.delete()
        }
    }
}
