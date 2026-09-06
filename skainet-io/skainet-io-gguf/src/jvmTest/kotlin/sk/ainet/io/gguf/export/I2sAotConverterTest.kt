package sk.ainet.io.gguf.export

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.SyntheticGguf
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertIs

/**
 * #1207: [I2sAotConverter] converts a `GROUP_128` (BitNet.cpp) I2_S tensor into a
 * `SEQUENTIAL`, trailer-scaled one — the shape #1203's loader serves as true zero-copy mmap —
 * while a non-I2_S tensor and the file's KV metadata pass through unchanged. The converted
 * file's decoded values must equal the source file's, and it must actually take the mmap fast
 * path afterwards, proving this is a real substitute for #1204's on-device sidecar cache, not
 * just a same-cost repack moved earlier.
 */
class I2sAotConverterTest {

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

    private fun sourceFile(): File {
        val elements = 256
        val codes = randomCodes(elements, seed = 9)
        val i2s = groupPayload(codes, qk = 128) + leFloat(0.5f) // BitNet.cpp GROUP_128 + trailer
        return SyntheticGguf.write(
            SyntheticGguf.TestTensor("attn.w", GGMLQuantizationType.I2_S, elements.toLong(), i2s),
            SyntheticGguf.tensor("attn.q4_0", GGMLQuantizationType.Q4_0, elements = 256, seed = 3),
        )
    }

    private fun convertedFile(source: File): File {
        val request = I2sAotConverter.convert(
            source = JvmRandomAccessSource.open(source),
            sourceLayout = I2sGgufLayout.GROUP_128,
        )
        val (_, bytes) = GGUFWriter.writeToByteArray(request)
        val out = Files.createTempFile("skainet-aot-converted", ".gguf")
        Files.write(out, bytes)
        return out.toFile()
    }

    private fun load(f: File, layout: I2sGgufLayout, form: WeightForm): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
                i2sLayout = layout,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun convertedFileDecodesIdenticallyAndTakesTheMmapFastPath() {
        val source = sourceFile()
        try {
            val converted = convertedFile(source)
            try {
                val fromSource = load(source, I2sGgufLayout.GROUP_128, WeightForm(residency = WeightResidency.HEAP))
                val fromConverted = load(converted, I2sGgufLayout.SEQUENTIAL, WeightForm(residency = WeightResidency.MAPPED))

                val sourceI2s = assertIs<BitNetB158TensorData>(fromSource.getValue("attn.w").data)
                val convertedI2s = assertIs<BitNetB158TensorData>(fromConverted.getValue("attn.w").data)
                assertContentEquals(
                    sourceI2s.toFloatArray(), convertedI2s.toFloatArray(),
                    "converted I2_S values must equal the source's",
                )
                assertIs<Storage.OffHeap>(
                    convertedI2s.packedStorage,
                    "the converted file's I2_S tensor must take the mmap fast path — no repack, no heap copy",
                )

                val sourceQ4 = fromSource.getValue("attn.q4_0").data as PackedBlockStorage
                val convertedQ4 = fromConverted.getValue("attn.q4_0").data as PackedBlockStorage
                assertContentEquals(
                    sourceQ4.toFloatArray(), convertedQ4.toFloatArray(),
                    "a passthrough (non-I2_S) tensor must be unchanged by conversion",
                )
            } finally {
                converted.delete()
            }
        } finally {
            source.delete()
        }
    }
}
