package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * #1098 (#973 census contradiction #6): GGUF writes dimensions in `ne` order, so a weight the
 * engine calls `[out, in]` arrives labelled `[in, out]` — while its bytes are already `[out, in]`
 * row-major. Only the label is wrong, and the label is what the block relayout reads.
 *
 * The guard that refuses a wrongly-labelled weight lives with the relayout, and is tested in
 * `PackedWeightsTest`.
 */
class WeightOrientationTest {

    /** A Q8_0 weight whose two dimensions differ and where only one of them is block-aligned. */
    private fun file(): File = SyntheticGguf.write(
        // ne = [in=128, out=3] → 384 elements, 12 blocks of 32
        SyntheticGguf.tensor("blk.0.attn_q.weight", GGMLQuantizationType.Q8_0, elements = 384),
    )

    private fun load(f: File, form: WeightForm): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val out = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
            ).load<FP32, Float>(ctx, FP32::class) { name, t -> out[name] = t }
        }
        return out
    }

    @Test
    fun `the default still reports the file's own ne order`() {
        val f = SyntheticGguf.write(SyntheticGguf.tensor("w", GGMLQuantizationType.F32, elements = 12))
        try {
            // a 1-D tensor has no orientation to get wrong, and nothing changes for it either way
            assertEquals(Shape(12), load(f, WeightForm()).getValue("w").shape)
            assertEquals(Shape(12), load(f, WeightForm(shape = WeightShapeOrientation.OUT_IN)).getValue("w").shape)
        } finally {
            f.delete()
        }
    }

    @Test
    fun `OUT_IN reverses a 2-D weight's label and nothing else`() {
        val f = twoDimensionalFile()
        try {
            val asStored = load(f, WeightForm()).getValue("w")
            val outIn = load(f, WeightForm(shape = WeightShapeOrientation.OUT_IN)).getValue("w")
            assertEquals(Shape(64, 4), asStored.shape, "ne order: [in, out]")
            assertEquals(Shape(4, 64), outIn.shape, "logical order: [out, in]")
            assertContentEquals(
                asStored.data.copyToFloatArray(), outIn.data.copyToFloatArray(),
                "the bytes are identical — only the label differs",
            )
        } finally {
            f.delete()
        }
    }

    /** `ne = [64, 4]`: 256 elements, 8 blocks of 32 — `in = 64`, `out = 4`. */
    private fun twoDimensionalFile(): File {
        val t = SyntheticGguf.tensor("w", GGMLQuantizationType.Q8_0, elements = 256)
        return SyntheticGguf.write(t.copy(dims = listOf(64L, 4L)))
    }
}
