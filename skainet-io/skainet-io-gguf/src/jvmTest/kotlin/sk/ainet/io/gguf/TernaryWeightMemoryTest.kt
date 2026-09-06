package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * The memory proof point of #1136 (evidence artifact for #1141): the same I2_S GGUF tensor loaded
 * keep-packed vs FP32-widened, byte counts compared. The packed path must hold the ~16× ratio —
 * 0.25 bytes per weight plus one FP32 scale against 4 bytes per weight.
 */
class TernaryWeightMemoryTest {

    private fun i2sTensor(name: String, elements: Int, seed: Int): SyntheticGguf.TestTensor {
        val rng = Random(seed)
        val qk = 128
        val bytesPerBlock = qk / 4
        val payload = ByteArray(elements / 4)
        for (j in 0 until elements) {
            val jb = j % qk
            val idx = (j / qk) * bytesPerBlock + jb % bytesPerBlock
            payload[idx] = (payload[idx].toInt() or (rng.nextInt(3) shl (6 - 2 * (jb / bytesPerBlock)))).toByte()
        }
        val trailer = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        repeat(8) { trailer.putFloat(0.5f) }
        return SyntheticGguf.TestTensor(name, GGMLQuantizationType.I2_S, elements.toLong(), payload + trailer.array())
    }

    private fun load(file: File, form: WeightForm?): Tensor<FP32, Float> {
        val ctx = DefaultDataExecutionContext()
        var tensor: Tensor<FP32, Float>? = null
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) },
                weightForm = form,
            ).load<FP32, Float>(ctx, FP32::class) { _, t -> tensor = t }
        }
        return tensor!!
    }

    @Test
    fun packedLoadIsAtLeastFifteenTimesSmallerThanTheWidenedLoad() {
        val elements = 2560 * 64 // one BitNet-2B-ish projection slice
        val file = SyntheticGguf.write(i2sTensor("w", elements, seed = 7))
        try {
            val packed = assertIs<BitNetB158TensorData>(load(file, form = null).data)
            val widened = assertIs<FloatArrayTensorData<*>>(
                load(file, WeightForm(encoding = EncodingRequest.DequantizeTo(FP32))).data,
            )
            val packedBytes = packed.packedData.size.toLong()
            val widenedBytes = widened.buffer.size.toLong() * 4
            val ratio = widenedBytes.toDouble() / packedBytes
            println(
                "[ternary-memory] $elements weights: packed=$packedBytes B, " +
                    "fp32-widened=$widenedBytes B, ratio=%.1fx".format(ratio),
            )
            assertTrue(ratio >= 15.0, "expected ~16x memory saving, measured %.1fx".format(ratio))
        } finally {
            file.delete()
        }
    }
}
