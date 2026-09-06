package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertIs

/**
 * #1203, second half: a `SEQUENTIAL`-layout I2_S tensor whose on-disk bytes are already a
 * complete `BITNET_B1_58` buffer (payload + its own trailing FP32 scale, no companion tensor to
 * override it) must load through [sk.ainet.io.JvmMappedFile]'s mmap branch — zero heap bytes,
 * zero repack — while every other case keeps today's heap-staged repack path.
 */
class I2sMappedFastPathTest {

    private fun sequentialPayload(codes: IntArray): ByteArray {
        val out = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            out[j / 4] = (out[j / 4].toInt() or (codes[j] shl ((j % 4) * 2))).toByte()
        }
        return out
    }

    private fun groupPayload(codes: IntArray, qk: Int): ByteArray {
        val bytesPerBlock = qk / 4
        val out = ByteArray(codes.size / 4)
        for (j in codes.indices) {
            val jb = j % qk
            val byteIndex = (j / qk) * bytesPerBlock + jb % bytesPerBlock
            out[byteIndex] = (out[byteIndex].toInt() or (codes[j] shl (6 - 2 * (jb / bytesPerBlock)))).toByte()
        }
        return out
    }

    private fun leFloat(value: Float): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array()

    private fun randomCodes(count: Int, seed: Int): IntArray {
        val rng = Random(seed)
        return IntArray(count) { rng.nextInt(3) } // {0, 1, 2} — never 3
    }

    private fun load(
        f: File,
        layout: I2sGgufLayout,
        form: WeightForm,
    ): Tensor<FP32, Float> {
        val ctx = DefaultDataExecutionContext()
        var tensor: Tensor<FP32, Float>? = null
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
                i2sLayout = layout,
            ).load<FP32, Float>(ctx, FP32::class) { _, t -> tensor = t }
        }
        return tensor!!
    }

    @Test
    fun sequentialWithATrailerAndNoCompanionMapsDirectlyOffTheFile() {
        val elements = 256
        val codes = randomCodes(elements, seed = 1)
        val bytes = sequentialPayload(codes) + leFloat(0.25f)
        val f = SyntheticGguf.write(SyntheticGguf.TestTensor("w", GGMLQuantizationType.I2_S, elements.toLong(), bytes))
        try {
            val mapped = load(f, I2sGgufLayout.SEQUENTIAL, WeightForm(residency = WeightResidency.MAPPED))
            val packed = assertIs<BitNetB158TensorData>(mapped.data)
            assertIs<Storage.OffHeap>(packed.packedStorage, "must be served over the file mapping, not a heap copy")

            val heap = load(f, I2sGgufLayout.SEQUENTIAL, WeightForm(residency = WeightResidency.HEAP))
            assertContentEquals(
                (heap.data as BitNetB158TensorData).toFloatArray(),
                packed.toFloatArray(),
                "mapped values must equal the heap-staged load",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun sequentialWithACompanionScaleStillRepacksOnHeap() {
        // The companion wins over a trailer for SEQUENTIAL (resolveI2sScale's own order), so even
        // though a trailer-shaped tail exists here too, the mmap fast path must not fire — its
        // bytes wouldn't be the real scale.
        val elements = 256
        val codes = randomCodes(elements, seed = 2)
        val payload = sequentialPayload(codes)
        val f = SyntheticGguf.write(
            SyntheticGguf.TestTensor("w", GGMLQuantizationType.I2_S, elements.toLong(), payload),
            SyntheticGguf.TestTensor("w_scale", GGMLQuantizationType.F32, 1L, leFloat(4f)), // divide-by convention
        )
        try {
            val mapped = load(f, I2sGgufLayout.SEQUENTIAL, WeightForm(residency = WeightResidency.MAPPED))
            val packed = assertIs<BitNetB158TensorData>(mapped.data)
            assertIs<Storage.Heap>(packed.packedStorage, "a companion-scale tensor must still repack, not mmap")
        } finally {
            f.delete()
        }
    }

    @Test
    fun group128NeverTakesTheMmapFastPathRegardlessOfTheTrailer() {
        // GROUP_128 payload bytes are not BITNET_B1_58-ordered — the mmap fast path must never
        // fire for it, trailer or not, since the raw bytes would be wrong without a repack.
        val elements = 256
        val codes = randomCodes(elements, seed = 3)
        val bytes = groupPayload(codes, qk = 128) + leFloat(0.5f)
        val f = SyntheticGguf.write(SyntheticGguf.TestTensor("w", GGMLQuantizationType.I2_S, elements.toLong(), bytes))
        try {
            val mapped = load(f, I2sGgufLayout.GROUP_128, WeightForm(residency = WeightResidency.MAPPED))
            val packed = assertIs<BitNetB158TensorData>(mapped.data)
            assertIs<Storage.Heap>(packed.packedStorage, "GROUP_128 must always repack, never mmap directly")
        } finally {
            f.delete()
        }
    }
}
