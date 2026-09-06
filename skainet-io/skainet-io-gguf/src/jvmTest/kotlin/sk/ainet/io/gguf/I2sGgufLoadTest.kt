package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.TernaryCodec
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
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * #1140 end-to-end: an I2_S GGUF loads **packed** — `BitNetB158TensorData`, 0.25 bytes per
 * weight — with the right values, for both converter flavors:
 *
 * - **BitNet.cpp** (group payload, FP32 scale in a 32-byte trailer after the payload;
 *   `w = (code−1)·scale`)
 * - **NeoGPU** (sequential payload, companion `<name>_scale` F32 scalar defined as "divide the
 *   output by it")
 *
 * The first ternary GGUF type that does not take the #1033 FP32 widening.
 */
class I2sGgufLoadTest {

    private fun randomCodes(count: Int, seed: Int): IntArray {
        val rng = Random(seed)
        return IntArray(count) { rng.nextInt(3) }
    }

    /** BitNet.cpp's `quantize_i2_s` packing + 32-byte trailer whose first 4 bytes are the scale. */
    private fun bitnetCppTensor(name: String, codes: IntArray, qk: Int, scale: Float): SyntheticGguf.TestTensor {
        val bytesPerBlock = qk / 4
        val payload = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            val jb = j % qk
            val byteIndex = (j / qk) * bytesPerBlock + jb % bytesPerBlock
            payload[byteIndex] = (payload[byteIndex].toInt() or (codes[j] shl (6 - 2 * (jb / bytesPerBlock)))).toByte()
        }
        val trailer = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        repeat(8) { trailer.putFloat(scale) } // BitNet.cpp's converter tiles the scale ×8
        return SyntheticGguf.TestTensor(name, GGMLQuantizationType.I2_S, codes.size.toLong(), payload + trailer.array())
    }

    /** NeoGPU's sequential packing (payload only) + its companion `<name>_scale` scalar. */
    private fun neogpuTensors(name: String, codes: IntArray, weightScale: Float): List<SyntheticGguf.TestTensor> {
        val payload = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            payload[j / 4] = (payload[j / 4].toInt() or (codes[j] shl ((j % 4) * 2))).toByte()
        }
        val scaleBytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(weightScale).array()
        return listOf(
            SyntheticGguf.TestTensor(name, GGMLQuantizationType.I2_S, codes.size.toLong(), payload),
            SyntheticGguf.TestTensor("${name}_scale", GGMLQuantizationType.F32, 1L, scaleBytes),
        )
    }

    private fun load(
        file: File,
        layout: I2sGgufLayout,
        form: WeightForm? = null,
    ): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) },
                weightForm = form,
                i2sLayout = layout,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun bitnetCppFlavorLoadsPackedWithTheTrailerScale() {
        val codes = randomCodes(256, seed = 1)
        val scale = 0.125f
        val file = SyntheticGguf.write(bitnetCppTensor("w", codes, qk = 128, scale = scale))
        try {
            val loaded = load(file, I2sGgufLayout.GROUP_128)
            assertEquals(setOf("w"), loaded.keys)
            val data = assertIs<BitNetB158TensorData>(loaded.getValue("w").data)
            assertEquals(scale, data.scale, "trailer scale folded into the BITNET_B1_58 buffer")
            assertEquals(256 / 4 + 4, data.packedData.size, "0.25 B/weight + 4 B scale — not widened")
            val expected = FloatArray(256) { (codes[it] - 1) * scale }
            assertContentEquals(expected, data.toFloatArray(), "values survive the group→sequential repack")
        } finally {
            file.delete()
        }
    }

    @Test
    fun neogpuFlavorLoadsPackedWithTheInverseCompanionScale() {
        val codes = randomCodes(128, seed = 2)
        val weightScale = 4.0f // NeoGPU semantics: divide output by it → multiplier 0.25
        val file = SyntheticGguf.write(*neogpuTensors("w", codes, weightScale).toTypedArray())
        try {
            val loaded = load(file, I2sGgufLayout.SEQUENTIAL)
            assertEquals(setOf("w"), loaded.keys, "the companion _scale tensor is consumed, not delivered")
            val data = assertIs<BitNetB158TensorData>(loaded.getValue("w").data)
            assertEquals(0.25f, data.scale)
            val expected = FloatArray(128) { (codes[it] - 1) * 0.25f }
            assertContentEquals(expected, data.toFloatArray())
        } finally {
            file.delete()
        }
    }

    @Test
    fun dequantizeToFp32StillWorksAndMatchesThePackedDecode() {
        val codes = randomCodes(256, seed = 3)
        val file = SyntheticGguf.write(bitnetCppTensor("w", codes, qk = 128, scale = 0.5f))
        try {
            val widened = load(file, I2sGgufLayout.GROUP_128, WeightForm(encoding = EncodingRequest.DequantizeTo(FP32)))
            val dense = assertIs<FloatArrayTensorData<*>>(widened.getValue("w").data)
            val expected = FloatArray(256) { (codes[it] - 1) * 0.5f }
            assertContentEquals(expected, dense.buffer.copyOf(256))
        } finally {
            file.delete()
        }
    }

    @Test
    fun group64FlavorRoundTrips() {
        val codes = randomCodes(128, seed = 4)
        val file = SyntheticGguf.write(bitnetCppTensor("w", codes, qk = 64, scale = 1.0f))
        try {
            val data = assertIs<BitNetB158TensorData>(load(file, I2sGgufLayout.GROUP_64).getValue("w").data)
            assertContentEquals(FloatArray(128) { (codes[it] - 1).toFloat() }, data.toFloatArray())
        } finally {
            file.delete()
        }
    }

    @Test
    fun theLoaderAndTheCodecAgreeOnTheRepackedBytes() {
        // The invariant that makes the kernels safe: whatever the loader emits decodes identically
        // through TernaryCodec — the reference the ternary kernel pack (#1138) is defined against.
        val codes = randomCodes(512, seed = 5)
        val file = SyntheticGguf.write(bitnetCppTensor("w", codes, qk = 128, scale = 0.75f))
        try {
            val data = assertIs<BitNetB158TensorData>(load(file, I2sGgufLayout.GROUP_128).getValue("w").data)
            assertContentEquals(
                TernaryCodec.decodeBitNet(data.packedData, 512),
                data.toFloatArray(),
            )
            assertTrue(data.packedData.size < 512, "packed, not widened")
        } finally {
            file.delete()
        }
    }
}
