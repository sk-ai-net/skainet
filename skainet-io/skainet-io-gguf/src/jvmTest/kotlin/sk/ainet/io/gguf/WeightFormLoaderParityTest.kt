package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1115/#1159: `WeightForm` is the loader's whole configuration surface. The flags-vs-form parity
 * matrix this file used to hold retired with the flags themselves; what remains pins the form
 * axes the loader validates and honours.
 */
class WeightFormLoaderParityTest {

    /** Mixed encodings, and a 2-D weight so the shape axis has something to reverse. */
    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 768)
            .copy(dims = listOf(256L, 3L)),
        SyntheticGguf.tensor("w_f16", GGMLQuantizationType.F16, elements = 1024),
    )

    private fun loadVia(f: File, build: (() -> JvmRandomAccessSource) -> StreamingGgufParametersLoader):
        Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            build { JvmRandomAccessSource.open(f) }
                .load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun `the default loader is the default form`() {
        val f = file()
        try {
            val implicit = loadVia(f) { src -> StreamingGgufParametersLoader(sourceProvider = src) }
            val explicit = loadVia(f) { src ->
                StreamingGgufParametersLoader(sourceProvider = src, weightForm = WeightForm.AS_STORED_ON_HEAP)
            }
            for ((name, tensor) in implicit) {
                assertContentEquals(
                    tensor.data.copyToFloatArray(), explicit.getValue(name).data.copyToFloatArray(), name,
                )
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `KERNEL_FEED is accepted now that packed storage can declare its order`() {
        // This slice refused it: packed TensorData read packedData as canonical row-major, so
        // feed-order bytes decoded to the wrong elements silently. #1120 gave the bytes a way to
        // say what order they are in, so the refusal is gone — replaced by the one constraint that
        // remains real, since feed order is defined relative to an [out, in] weight.
        StreamingGgufParametersLoader(
            sourceProvider = { JvmRandomAccessSource.open(file()) },
            weightForm = WeightForm(
                order = WeightByteOrder.KERNEL_FEED,
                shape = WeightShapeOrientation.OUT_IN,
            ),
        )

        val failure = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(order = WeightByteOrder.KERNEL_FEED),
            )
        }
        assertTrue(failure.message!!.contains("OUT_IN"), failure.message!!)
    }

    @Test
    fun `an encoding request this loader cannot honour is refused up front`() {
        val requantize = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(
                    encoding = EncodingRequest.RequantizeTo(sk.ainet.lang.tensor.storage.TensorEncoding.Q8_0),
                ),
            )
        }
        assertTrue(requantize.message!!.contains("quantizer"), requantize.message!!)

        val toFp16 = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file()) },
                weightForm = WeightForm(
                    encoding = EncodingRequest.DequantizeTo(sk.ainet.lang.types.FP16),
                ),
            )
        }
        assertTrue(toFp16.message!!.contains("FP32 only"), toFp16.message!!)
    }
}
