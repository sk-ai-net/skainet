package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * #1094 (#973): which block order a packed weight is in is a property of the value.
 *
 * The two orders coincide only when a row is a single block, so every assertion here uses a weight
 * that is **three** blocks wide — the case where mixing them produces a block-permuted matrix of
 * plausible, finite, wrong numbers. That is the failure #968/#971 shipped, and the reason the order
 * is now carried rather than guessed.
 */
class BlockOrderTest {

    private val rows = 4
    private val blocksPerRow = 3
    private val blockSize = 32
    private val bytesPerBlock = 34
    private val cols = blocksPerRow * blockSize

    /** A Q8_0 weight whose every block is identifiable: block (o, b) is filled with `o * 16 + b`. */
    private fun weight(): TensorView {
        val bytes = ByteArray(rows * blocksPerRow * bytesPerBlock)
        for (o in 0 until rows) {
            for (b in 0 until blocksPerRow) {
                val base = (o * blocksPerRow + b) * bytesPerBlock
                bytes[base] = 0x00; bytes[base + 1] = 0x3C        // fp16 scale 1.0
                for (i in 0 until blockSize) bytes[base + 2 + i] = (o * 16 + b).toByte()
            }
        }
        val data = Q8_0BlockTensorData(Shape(rows, cols), bytes)
        return TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(rows, cols), TensorEncoding.Q8_0, PackedBlockDecoder(data),
        )
    }

    @Test
    fun aViewLoadedFromAFileIsRowMajor() {
        val w = weight()
        assertEquals(BlockOrder.ROW_MAJOR, w.layout.blockOrder, "canonical is the default, as a file holds it")
        assertTrue(w.layout.blocked)
        assertTrue(w.layout.toString().contains("row_major"), w.layout.toString())
    }

    @Test
    fun prepackingChangesTheBytesButNotTheMatrix() {
        val canonical = weight()
        val kernelOrder = canonical.prepack(BlockOrder.INPUT_BLOCK_MAJOR)

        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, kernelOrder.layout.blockOrder)
        assertNotEquals(canonical.storage.id, kernelOrder.storage.id, "a relayout is a copy, not a view")

        // the decoded matrix is identical — get() and toFloatArray() read through the order
        assertContentEquals(canonical.toFloatArray(), kernelOrder.toFloatArray(), "the matrix must not change")
        for (r in 0 until rows) for (c in 0 until cols) {
            assertEquals(canonical.get(r, c), kernelOrder.get(r, c), "element ($r,$c)")
        }
    }

    @Test
    fun theBytesActuallyMoveToKernelFeedOrder() {
        val kernelOrder = weight().prepack(BlockOrder.INPUT_BLOCK_MAJOR)
        val bytes = (kernelOrder.storage as Storage.Heap).bytes!!
        val offset = (kernelOrder.storage as Storage.Heap).arrayOffset
        // block (o, b) must now sit at flat index b * rows + o
        for (o in 0 until rows) {
            for (b in 0 until blocksPerRow) {
                val at = offset + (b * rows + o) * bytesPerBlock + 2
                assertEquals((o * 16 + b).toByte(), bytes[at], "block ($o,$b) should be at input-major index ${b * rows + o}")
            }
        }
    }

    @Test
    fun prepackingBackIsTheIdentity() {
        // the double-transpose hazard: on 0.40.1 `transpose(transpose(W)) != W` for a non-square
        // block grid and nothing detected it. A relayout that names its target order cannot have
        // that problem — going there and back is exactly the original.
        val canonical = weight()
        val roundTrip = canonical.prepack(BlockOrder.INPUT_BLOCK_MAJOR).prepack(BlockOrder.ROW_MAJOR)
        assertEquals(BlockOrder.ROW_MAJOR, roundTrip.layout.blockOrder)
        assertContentEquals(canonical.toFloatArray(), roundTrip.toFloatArray())
        val original = (canonical.storage as Storage.Heap).bytes!!
        val after = (roundTrip.storage as Storage.Heap).bytes!!
        assertContentEquals(original, after.copyOfRange((roundTrip.storage as Storage.Heap).arrayOffset, (roundTrip.storage as Storage.Heap).arrayOffset + original.size))
        assertTrue(blocksPerRow != rows, "this test is only meaningful on a non-square block grid")
    }

    @Test
    fun prepackingToTheOrderItAlreadyHasIsFree() {
        val w = weight()
        val sink = RecordingTraceSink()
        val same = w.prepack(BlockOrder.ROW_MAJOR, Scope.Ambient, sink)
        assertEquals(w.storage.id, same.storage.id, "no copy when nothing has to move")
        assertTrue(sink.eventsOf<TraceEvent.AdapterInserted>().isEmpty(), "and nothing to report")
    }

    @Test
    fun aRelayoutIsVisibleAndPriced() {
        val sink = RecordingTraceSink()
        val scope = ForwardScope(slabFloats = rows * blocksPerRow * bytesPerBlock, sink = sink, name = "prepack")
        weight().prepack(BlockOrder.INPUT_BLOCK_MAJOR, scope, sink)
        val adapter = sink.eventsOf<TraceEvent.AdapterInserted>().single()
        assertEquals("prepack-input_block_major", adapter.kind)
        assertEquals((rows * blocksPerRow * bytesPerBlock).toLong(), adapter.bytes, "the copy is priced in bytes")
        assertEquals(ScopeKind.FORWARD, adapter.scope)
        scope.close()
    }

    @Test
    fun theOrderSurvivesTheOperationsThatDoNotMoveBytes() {
        val kernelOrder = weight().prepack(BlockOrder.INPUT_BLOCK_MAJOR)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, kernelOrder.narrow(0, 1, 2).layout.blockOrder)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, kernelOrder.transpose().layout.blockOrder)
        assertEquals(BlockOrder.INPUT_BLOCK_MAJOR, kernelOrder.unsqueeze(0).layout.blockOrder)
    }

    @Test
    fun onlyAPackedViewHasABlockOrderToChange() {
        val dense = TensorView.dense(Storage.Heap.floats(16), Shape(4, 4))
        assertEquals(BlockOrder.ROW_MAJOR, dense.layout.blockOrder, "the default is meaningless but harmless for dense")
        assertFailsWith<IllegalArgumentException> { dense.prepack(BlockOrder.INPUT_BLOCK_MAJOR) }
    }
}
