package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * #1150: `RequantizeTo(BITNET_PLANES)` — the loader's one requantizer. A trained FP32 lm_head
 * weight (or an I2_S one) arrives as packed [BitNetPlanesTensorData], encoded once at load,
 * reconstructing within the format's truncation bound.
 */
class PlanesRequantizeLoadTest {

    private val planesForm = WeightForm(
        encoding = EncodingRequest.RequantizeTo(TensorEncoding.BITNET_PLANES),
        shape = WeightShapeOrientation.OUT_IN,
    )

    private fun f32Tensor(name: String, out: Int, inDim: Int, values: FloatArray): SyntheticGguf.TestTensor {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buf.putFloat(it) }
        // GGUF `ne` order is fastest-varying first: a [out, in] row-major weight declares [in, out].
        return SyntheticGguf.TestTensor(
            name, GGMLQuantizationType.F32, values.size.toLong(), buf.array(),
            dims = listOf(inDim.toLong(), out.toLong()),
        )
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

    @Test
    fun f32WeightRequantizesToPackedPlanesWithinTheTruncationBound() {
        val out = 6; val inDim = 32
        val rng = Random(3)
        val values = FloatArray(out * inDim) { (rng.nextFloat() - 0.5f) * 2f }
        val file = SyntheticGguf.write(f32Tensor("output.weight", out, inDim, values))
        try {
            val data = assertIs<BitNetPlanesTensorData>(load(file, planesForm).getValue("output.weight").data)
            assertEquals(out, data.rows)
            assertEquals(inDim, data.cols)
            val decoded = TernaryCodec.decodeBitNetPlanes(data.packedData, out, inDim)
            for (r in 0 until out) {
                val bound = data.rowScale(r) * (0.5f / 2187f) + 1e-4f
                for (c in 0 until inDim) {
                    val err = abs(values[r * inDim + c] - decoded[r * inDim + c])
                    assertTrue(err <= bound, "[$r,$c]: err=$err > $bound")
                }
            }
        } finally {
            file.delete()
        }
    }

    @Test
    fun requantizeToAnythingElseStillFailsEagerly() {
        assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { throw IllegalStateException("never opened") },
                weightForm = WeightForm(
                    encoding = EncodingRequest.RequantizeTo(TensorEncoding.TQ2_0),
                    shape = WeightShapeOrientation.OUT_IN,
                ),
            )
        }
    }

    @Test
    fun planesWithoutOutInOrientationFailsEagerly() {
        assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { throw IllegalStateException("never opened") },
                weightForm = WeightForm(encoding = EncodingRequest.RequantizeTo(TensorEncoding.BITNET_PLANES)),
            )
        }
    }
}
