package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * #1033 (M2-F1): a GGUF holding `TQ1_0` / `TQ2_0` tensors loads, and the values that come out are
 * the values that went in — the end-to-end half of the ternary-encoding slice.
 *
 * Before this slice both types decoded through a hand-written routine whose element order did not
 * match `dequantize_row_tq{1,2}_0` (four *consecutive* elements per byte instead of four 32 apart),
 * and the loader described them as `Opaque`, so nothing downstream knew their geometry.
 */
class GgufTernaryLoadTest {

    @Test
    fun ternaryTensorsLoadWithTheirValuesIntact() {
        val (tq1, _, tq1Values) = SyntheticGguf.ternary("w_tq1", GGMLQuantizationType.TQ1_0, elements = 512)
        val (tq2, _, tq2Values) = SyntheticGguf.ternary("w_tq2", GGMLQuantizationType.TQ2_0, elements = 512)
        val file = SyntheticGguf.write(tq1, tq2)
        try {
            val loaded = load(file, WeightForm(encoding = EncodingRequest.DequantizeTo(FP32)))
            assertEquals(setOf("w_tq1", "w_tq2"), loaded.keys)
            for ((name, expected) in listOf("w_tq1" to tq1Values, "w_tq2" to tq2Values)) {
                val data = loaded.getValue(name).data
                val actual = (data as FloatArrayTensorData<*>).buffer
                assertContentEquals(expected, actual.copyOf(expected.size), "$name: decoded values")
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun theHeaderDescribesTheTernaryGeometryInsteadOfCallingItOpaque() {
        val (tq1, _, _) = SyntheticGguf.ternary("w_tq1", GGMLQuantizationType.TQ1_0, elements = 512)
        val file = SyntheticGguf.write(tq1)
        try {
            JvmRandomAccessSource.open(file).use { src ->
                val reader = StreamingGGUFReader.open(src)
                val storage = reader.loadTensorStorage("w_tq1")
                assertEquals(TensorEncoding.TQ1_0, storage.encoding)
                assertEquals(FP32, storage.dtype, "ternary weights are logically FP32 (SKEEP-003 rule 3)")
                assertEquals(54L * 2, storage.encoding.physicalBytes(512), "two 54-byte blocks")
                assertEquals(1.625, storage.encoding.blockSpec!!.bitsPerElement)

                // the plan sees the same geometry the file has
                val input = reader.planInput(ctx = 128)
                assertEquals(54L * 2, input.weights.single { it.name == "w_tq1" }.bytes)
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun theLoaderAndTheCodecDecodeTheSameBytesIdentically() {
        val (tq2, bytes, values) = SyntheticGguf.ternary("w", GGMLQuantizationType.TQ2_0, elements = 256 * 3)
        val file = SyntheticGguf.write(tq2)
        try {
            val loaded = load(file, WeightForm(encoding = EncodingRequest.DequantizeTo(FP32))).getValue("w").data
            val actual = (loaded as FloatArrayTensorData<*>).buffer.copyOf(values.size)
            assertContentEquals(TernaryCodec.decodeTq2_0(bytes, values.size), actual, "loader vs reference codec")
            assertContentEquals(values, actual)
        } finally {
            file.delete()
        }
    }

    private fun load(file: File, form: WeightForm): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) },
                weightForm = form,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }
}
