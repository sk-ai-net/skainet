package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.math.abs
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1120: a weight loaded in kernel feed order says so, decodes to the same matrix, and reaches the
 * kernels without the per-weight relayout ever running.
 *
 * Feed order is only meaningful at three-plus blocks per row — at one block per row the two orders
 * coincide and every assertion below would hold vacuously (#968).
 */
class KernelFeedOrderTest {

    private val outDim = 32
    private val inDim = 96          // three Q8_0 blocks per row

    @BeforeTest
    fun registerKernels() {
        KernelRegistry.register(ScalarKernelProvider)
    }

    private fun modelFile(type: GGMLQuantizationType = GGMLQuantizationType.Q8_0): File =
        SyntheticGguf.write(
            SyntheticGguf.tensor("blk.0.attn_q.weight", type, elements = outDim * inDim)
                .copy(dims = listOf(inDim.toLong(), outDim.toLong())),
        )

    private fun load(f: File, form: WeightForm, sink: RecordingTraceSink = RecordingTraceSink()):
        Pair<Tensor<FP32, Float>, RecordingTraceSink> {
        val ctx = DirectCpuExecutionContext()
        var w: Tensor<FP32, Float>? = null
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
                traceSink = sink,
            ).load<FP32, Float>(ctx, FP32::class) { _, t -> w = t }
        }
        return w!! to sink
    }

    private fun asStored() = WeightForm(shape = WeightShapeOrientation.OUT_IN)
    private fun feedOrder() =
        WeightForm(order = WeightByteOrder.KERNEL_FEED, shape = WeightShapeOrientation.OUT_IN)

    @Test
    fun `a feed-order weight declares its order and keeps its shape`() {
        val f = modelFile()
        try {
            val (w, _) = load(f, feedOrder())
            val packed = w.data as PackedBlockStorage
            assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, packed.blockOrder, "the bytes must say what order they are in")
            assertEquals(Shape(outDim, inDim), w.shape, "feed order is a property of the bytes, not of the shape")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `feed-order bytes decode to the same matrix they were permuted from`() {
        val f = modelFile()
        try {
            val (canonical, _) = load(f, asStored())
            val (feed, _) = load(f, feedOrder())

            val canonicalBytes = (canonical.data as PackedBlockStorage).packedData
            val feedBytes = (feed.data as PackedBlockStorage).packedData
            assertTrue(
                !canonicalBytes.contentEquals(feedBytes),
                "at three blocks per row the two orders must differ, or this test proves nothing",
            )

            assertContentEquals(
                (canonical.data as PackedBlockStorage).toFloatArray(),
                (feed.data as PackedBlockStorage).toFloatArray(),
                "different bytes, same matrix — that is the whole claim of a declared block order",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the product is the same whichever order the weight arrived in`() {
        val f = modelFile()
        try {
            val ctx = DirectCpuExecutionContext()
            val xs = FloatArray(inDim) { (it % 13) * 0.0625f }
            val x = ctx.fromFloatArray<FP32, Float>(Shape(1, inDim), FP32::class, xs)

            val (canonical, _) = load(f, asStored())
            val (feed, _) = load(f, feedOrder())

            val fromCanonical = x.matmulWeightTransposed(canonical).data.copyToFloatArray()
            val fromFeed = x.matmulWeightTransposed(feed).data.copyToFloatArray()

            for (o in fromCanonical.indices) {
                assertTrue(
                    abs(fromCanonical[o] - fromFeed[o]) <= 1e-3f * maxOf(1.0f, abs(fromCanonical[o])),
                    "output[$o]: canonical ${fromCanonical[o]} vs feed-order ${fromFeed[o]}",
                )
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the permutation happens once at load and is reported there`() {
        val f = modelFile()
        try {
            val (_, sink) = load(f, feedOrder())
            val relayouts = sink.events()
                .filterIsInstance<TraceEvent.AdapterInserted>()
                .filter { it.kind.startsWith("prepack") }
            assertEquals(1, relayouts.size, "one weight, one permutation: ${sink.events().map { it }}")

            val (_, quiet) = load(f, asStored())
            assertTrue(
                quiet.events().filterIsInstance<TraceEvent.AdapterInserted>().none { it.kind.startsWith("prepack") },
                "a weight kept as stored is not permuted",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `feed order without OUT_IN is refused because it would be meaningless`() {
        val failure = assertFailsWith<IllegalArgumentException> {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(modelFile()) },
                weightForm = WeightForm(order = WeightByteOrder.KERNEL_FEED),
            )
        }
        assertTrue(failure.message!!.contains("OUT_IN"), failure.message!!)
    }
}
