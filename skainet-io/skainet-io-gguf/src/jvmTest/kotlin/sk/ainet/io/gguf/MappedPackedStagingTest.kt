package sk.ainet.io.gguf

import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BufferPackedTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * #1189: under `WeightResidency.MAPPED`, Q4_K/Q6_K payloads must be served as
 * [BufferPackedTensorData] — a row-major view over the file mapping, zero heap bytes — while
 * every other type keeps its pre-#1189 staging (heap packed data / mapped dense F32). Values
 * must be identical to the heap load; StagingPolicyParityTest asserts that across policies,
 * this test pins the *representation*.
 */
class MappedPackedStagingTest {

    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("w_q6k", GGMLQuantizationType.Q6_K, elements = 1024),
        SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 1024),
    )

    private fun load(f: File, form: WeightForm): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun `mapped staging keeps k-quant payloads off the heap, row-major`() {
        val f = file()
        try {
            val mapped = load(f, WeightForm(residency = WeightResidency.MAPPED))

            val q4k = mapped.getValue("w_q4k").data
            assertTrue(q4k is BufferPackedTensorData, "Q4_K under MAPPED must be buffer-packed, got ${q4k::class.simpleName}")
            assertEquals(TensorEncoding.Q4_K, q4k.encoding)
            assertEquals(BlockOrder.ROW_MAJOR, q4k.blockOrder)

            val q6k = mapped.getValue("w_q6k").data
            assertTrue(q6k is BufferPackedTensorData, "Q6_K under MAPPED must be buffer-packed, got ${q6k::class.simpleName}")

            // #1192: Q8_0 (the k-quant fallback format) is mapped-servable too.
            val q80 = mapped.getValue("w_q80").data
            assertTrue(q80 is BufferPackedTensorData, "Q8_0 under MAPPED is buffer-packed since #1192, got ${q80::class.simpleName}")

            // Values equal the heap load, element for element.
            val heap = load(f, WeightForm(residency = WeightResidency.HEAP))
            for (name in listOf("w_q4k", "w_q6k", "w_q80")) {
                assertContentEquals(
                    heap.getValue(name).data.copyToFloatArray(),
                    mapped.getValue(name).data.copyToFloatArray(),
                    "values of $name",
                )
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `heap residency is untouched`() {
        val f = file()
        try {
            val heap = load(f, WeightForm(residency = WeightResidency.HEAP))
            val q4k = heap.getValue("w_q4k").data
            assertTrue(q4k is Q4_KBlockTensorData, "HEAP residency keeps the heap tensor data")
            assertTrue((q4k as PackedBlockStorage).packedData.isNotEmpty())
        } finally {
            f.delete()
        }
    }
}
