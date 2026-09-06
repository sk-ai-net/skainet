package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.plan.PlannerProfile
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
 * #1202: an I2_S payload at least [PlannerProfile.OFF_HEAP_THRESHOLD] bytes loads into off-heap
 * storage instead of a permanent `ByteArray` — the fix for #1198's ART-heap-cap OOM risk — and
 * decodes to exactly the same values a heap-backed load would produce.
 */
class I2sOffHeapLoadTest {

    /** BitNet.cpp's GROUP_128 packing + 32-byte trailer, sized to cross the off-heap threshold. */
    private fun bitnetCppTensor(name: String, elements: Int, seed: Int, scale: Float): SyntheticGguf.TestTensor {
        val rng = Random(seed)
        val qk = 128
        val bytesPerBlock = qk / 4
        val payload = ByteArray(elements / 4)
        for (j in 0 until elements) {
            val jb = j % qk
            val byteIndex = (j / qk) * bytesPerBlock + jb % bytesPerBlock
            payload[byteIndex] = (payload[byteIndex].toInt() or (rng.nextInt(3) shl (6 - 2 * (jb / bytesPerBlock)))).toByte()
        }
        val trailer = ByteBuffer.allocate(32).order(ByteOrder.LITTLE_ENDIAN)
        repeat(8) { trailer.putFloat(scale) }
        return SyntheticGguf.TestTensor(name, GGMLQuantizationType.I2_S, elements.toLong(), payload + trailer.array())
    }

    private fun load(file: File): Tensor<FP32, Float> {
        val ctx = DefaultDataExecutionContext()
        var tensor: Tensor<FP32, Float>? = null
        runBlocking {
            StreamingGgufParametersLoader(sourceProvider = { JvmRandomAccessSource.open(file) })
                .load<FP32, Float>(ctx, FP32::class) { _, t -> tensor = t }
        }
        return tensor!!
    }

    @Test
    fun payloadAtTheThresholdLoadsOffHeapAndDecodesCorrectly() {
        // payload = elements / 4 bytes; pick elements so the payload lands exactly at the
        // threshold (a multiple of 128 for GROUP_128 blocking).
        val elements = (PlannerProfile.OFF_HEAP_THRESHOLD * 4).toInt()
        val file = SyntheticGguf.write(bitnetCppTensor("w", elements, seed = 3, scale = 0.25f))
        try {
            val packed = assertIs<BitNetB158TensorData>(load(file).data)
            assertIs<Storage.OffHeap>(packed.packedStorage, "a $elements-element I2_S payload must not be a permanent heap array")

            // The lazily-materialized snapshot must equal exactly what was written into
            // off-heap storage — i.e. copyFrom (write) and copyInto (read) round-trip correctly
            // through the real load path, not just in the Storage unit tests.
            assertContentEquals(
                TernaryCodec.decodeBitNet(packed.packedData, elements),
                packed.toFloatArray(),
            )
        } finally {
            file.delete()
        }
    }

    @Test
    fun payloadBelowTheThresholdStaysOnHeap() {
        val elements = 256 // well under the threshold
        val file = SyntheticGguf.write(bitnetCppTensor("w", elements, seed = 3, scale = 0.25f))
        try {
            val packed = assertIs<BitNetB158TensorData>(load(file).data)
            assertIs<Storage.Heap>(packed.packedStorage, "a tiny I2_S payload should stay on the heap, not pay an off-heap allocation")
        } finally {
            file.delete()
        }
    }
}
