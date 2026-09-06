package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.PackedBlockDecoder
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1097 (#973.3): one implementation of the block relayout, owned by the engine, and fixtures a
 * downstream repository can assert against.
 *
 * The census in #973 found the same permutation reimplemented in several places, drifting; the
 * point of these tests is that there is now exactly one, that it is its own inverse, and that a
 * change to what "canonical" means fails a test rather than shipping.
 */
class PackedWeightsTest {

    private val rows = 4
    private val blocksPerRow = 3
    private val bytesPerBlock = 34          // Q8_0

    @Test
    fun theByteLevelRelayoutMovesEachBlockWhereTheContractSays() {
        val canonical = PackedLayoutFixtures.canonical(TensorEncoding.Q8_0)
        val kernelOrder = PackedWeights.toKernelOrder(canonical, rows, blocksPerRow, bytesPerBlock)
        assertNull(PackedLayoutFixtures.disagreement(canonical, TensorEncoding.Q8_0, kernelOrder = false))
        assertNull(PackedLayoutFixtures.disagreement(kernelOrder, TensorEncoding.Q8_0, kernelOrder = true))
        assertNotEquals(
            canonical.toList(), kernelOrder.toList(),
            "with three blocks per row the two orders must differ — at one block they coincide, which is what hid #968",
        )
    }

    @Test
    fun theRelayoutIsItsOwnInverse() {
        val canonical = PackedLayoutFixtures.canonical(TensorEncoding.Q4_K)
        val bytes = PackedLayoutFixtures.bytesPerBlockOf(TensorEncoding.Q4_K)
        val roundTrip = PackedWeights.toCanonicalOrder(
            PackedWeights.toKernelOrder(canonical, rows, blocksPerRow, bytes), rows, blocksPerRow, bytes,
        )
        assertContentEquals(canonical, roundTrip, "the permutation must be invertible — unlike the transpose it replaces")
    }

    @Test
    fun everyCoveredFormatHasAFixtureThatAgreesWithItsOwnDescriptor() {
        for (encoding in PackedLayoutFixtures.encodings) {
            val bytes = PackedLayoutFixtures.bytesPerBlockOf(encoding)
            assertEquals(
                bytes.toLong() * blocksPerRow, encoding.physicalBytes(blocksPerRow.toLong() * blockSizeOf(encoding)),
                "${encoding.name}: the fixture's block size must be the encoding's own",
            )
            assertNull(PackedLayoutFixtures.disagreement(PackedLayoutFixtures.canonical(encoding), encoding, kernelOrder = false), encoding.name)
            assertNull(PackedLayoutFixtures.disagreement(PackedLayoutFixtures.kernelOrder(encoding), encoding, kernelOrder = true), encoding.name)
        }
    }

    @Test
    fun aDisagreementIsReportedWithTheBlockThatMoved() {
        // what a downstream test sees when its converter emits the other order
        val wrong = PackedLayoutFixtures.kernelOrder(TensorEncoding.Q8_0)
        val message = PackedLayoutFixtures.disagreement(wrong, TensorEncoding.Q8_0, kernelOrder = false)
        assertTrue(message != null && message.contains("should be at flat index"), message ?: "no disagreement reported")
        assertTrue(message!!.contains("canonical"), message)
    }

    @Test
    fun prepackForMatmulIsIdempotent() {
        val bytes = PackedLayoutFixtures.canonical(TensorEncoding.Q8_0)
        val shape = Shape(rows, blocksPerRow * 32)
        val view = TensorView.packed(
            Storage.Heap.wrap(bytes), shape, TensorEncoding.Q8_0,
            PackedBlockDecoder(Q8_0BlockTensorData(shape, bytes)),
        )
        val once = PackedWeights.prepackForMatmul(view)
        val twice = PackedWeights.prepackForMatmul(once)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, once.layout.blockOrder)
        assertEquals(once.storage.id, twice.storage.id, "calling it twice must not copy again — it is not a transpose")
        assertContentEquals(view.toFloatArray(), once.toFloatArray(), "and the matrix is unchanged")

        val back = PackedWeights.toCanonical(once)
        assertEquals(BlockOrder.ROW_MAJOR, back.layout.blockOrder)
        assertContentEquals(view.toFloatArray(), back.toFloatArray())
    }

    @Test
    fun itRefusesWhatItCannotRelayout() {
        val dense = TensorView.dense(Storage.Heap.floats(16), Shape(4, 4))
        assertFailsWith<IllegalArgumentException> { PackedWeights.prepackForMatmul(dense) }
        assertFailsWith<IllegalArgumentException> { PackedWeights.blocksPerRow(TensorEncoding.Dense(4), 128) }
        assertFailsWith<IllegalArgumentException> {
            PackedWeights.blocksPerRow(TensorEncoding.Q8_0, 100)   // not a multiple of the block size
        }
        assertEquals(4, PackedWeights.blocksPerRow(TensorEncoding.Q8_0, 128))
        assertEquals(2, PackedWeights.blocksPerRow(TensorEncoding.Q4_K, 512))
    }


    @Test
    fun aWeightLabelledTheWrongWayRoundIsRefusedRatherThanMisPermuted() {
        // #1098 / #973 census #6: a packed weight's blocks tile the *input* dimension, so
        // relayouting an [in, out] label permutes the wrong grid — which is what a GGUF's ne order
        // hands you today.
        val failure = assertFailsWith<IllegalArgumentException> {
            PackedWeights.requireOutIn(rows = 128, inputDim = 3, encoding = TensorEncoding.Q8_0)
        }
        assertTrue(failure.message!!.contains("looks like [in, out]"), failure.message!!)
        assertTrue(failure.message!!.contains("WeightShapeOrientation.OUT_IN"), "and says how to fix it")

        PackedWeights.requireOutIn(rows = 3, inputDim = 128, encoding = TensorEncoding.Q8_0)
    }

    @Test
    fun theOrientationGuardStaysQuietWhenItCannotTell() {
        // both dimensions block-aligned: ambiguous, and a false refusal would be worse than none
        PackedWeights.requireOutIn(rows = 64, inputDim = 128, encoding = TensorEncoding.Q8_0)
        PackedWeights.requireOutIn(rows = 128, inputDim = 128, encoding = TensorEncoding.Q8_0)
        // an unaligned input dimension cannot be relayouted whichever way round it is
        assertFailsWith<IllegalArgumentException> {
            PackedWeights.requireOutIn(rows = 5, inputDim = 7, encoding = TensorEncoding.Q8_0)
        }
        // a dense encoding has no block grid, so there is nothing to check
        PackedWeights.requireOutIn(rows = 5, inputDim = 7, encoding = TensorEncoding.Dense(4))
    }

    private fun blockSizeOf(encoding: TensorEncoding): Int = when (encoding) {
        TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0 -> 32
        else -> 256
    }
}
