package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * #1120: a weight the loader already put in kernel feed order costs nothing to use.
 *
 * The point of declaring block order is not tidiness — it is that the O(bytes) relayout #1096 runs
 * once per weight stops running at all. Asserted structurally rather than by timing: the bytes the
 * kernels are handed must be *the same array*, not an equal one.
 */
class FeedOrderWeightNoCopyTest {

    private val ctx = DirectCpuExecutionContext()
    private val outDim = 32
    private val inDim = 96          // three blocks per row: the orders differ (#968)

    private fun bytes(): ByteArray {
        val out = ByteArray(outDim * (inDim / 32) * 34)
        var seed = 7
        for (b in 0 until outDim * (inDim / 32)) {
            val base = b * 34
            out[base] = 0x00; out[base + 1] = 0x3C
            for (i in 0 until 32) {
                seed = seed * 1103515245 + 12345
                out[base + 2 + i] = ((seed ushr 16) % 9 - 4).toByte()
            }
        }
        return out
    }

    @Suppress("UNCHECKED_CAST")
    private fun weight(order: BlockOrder, payload: ByteArray): Tensor<FP32, Float> =
        ctx.fromData(
            Q8_0BlockTensorData(Shape(outDim, inDim), payload, order) as TensorData<FP32, Float>,
            FP32::class,
        )

    @Test
    fun `a feed-order weight reaches the kernels without its bytes being copied`() {
        val payload = bytes()
        val w = weight(BlockOrder.INPUT_BLOCK_MAJOR, payload)

        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, FloatArray(inDim) { (it % 13) * 0.0625f })
        val result = x.matmulWeightTransposed(w)
        assertEquals(Shape(1, outDim), result.shape)

        // The claim: the same array, still. A relayout would have produced a different one.
        assertSame(payload, (w.data as PackedBlockStorage).packedData, "the weight's own bytes must be untouched")
    }

    @Test
    fun `a feed-order weight and the canonical weight it came from give the same product`() {
        val canonical = bytes()
        val canonicalWeight = weight(BlockOrder.ROW_MAJOR, canonical)

        // Permute canonically-ordered blocks into feed order by hand: block (o, b) moves from
        // o * blocksPerRow + b to b * rows + o.
        val blocksPerRow = inDim / 32
        val feed = ByteArray(canonical.size)
        for (o in 0 until outDim) {
            for (b in 0 until blocksPerRow) {
                canonical.copyInto(feed, (b * outDim + o) * 34, (o * blocksPerRow + b) * 34, (o * blocksPerRow + b + 1) * 34)
            }
        }
        val feedWeight = weight(BlockOrder.INPUT_BLOCK_MAJOR, feed)

        assertTrue(!canonical.contentEquals(feed), "the two orders must differ here or the test is vacuous")
        assertTrue(
            (canonicalWeight.data as PackedBlockStorage).toFloatArray()
                .contentEquals((feedWeight.data as PackedBlockStorage).toFloatArray()),
            "different bytes, same matrix",
        )

        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, FloatArray(inDim) { (it % 13) * 0.0625f })
        val fromCanonical = x.matmulWeightTransposed(canonicalWeight).data.copyToFloatArray()
        val fromFeed = x.matmulWeightTransposed(feedWeight).data.copyToFloatArray()
        for (o in fromCanonical.indices) {
            assertTrue(
                abs(fromCanonical[o] - fromFeed[o]) <= 1e-3f * maxOf(1.0f, abs(fromCanonical[o])),
                "output[$o]: canonical ${fromCanonical[o]} vs feed ${fromFeed[o]}",
            )
        }
    }
}
