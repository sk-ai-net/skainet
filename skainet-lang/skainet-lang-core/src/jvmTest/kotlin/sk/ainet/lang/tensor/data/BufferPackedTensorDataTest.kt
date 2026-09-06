package sk.ainet.lang.tensor.data

import java.nio.file.Files
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.DirectBufferStorage
import sk.ainet.lang.memory.MappedBufferStorage
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * #1189: a [BufferPackedTensorData] over off-heap bytes must decode exactly like the heap
 * `TensorData` for the same bytes — mapped file or direct buffer, Q4_K and Q6_K.
 */
class BufferPackedTensorDataTest {

    private fun randomPayload(numBlocks: Int, bytesPerBlock: Int, fp16At: IntArray, seed: Int): ByteArray {
        val rng = Random(seed)
        val bytes = ByteArray(numBlocks * bytesPerBlock)
        rng.nextBytes(bytes)
        for (block in 0 until numBlocks) {
            for (off in fp16At) {
                bytes[block * bytesPerBlock + off] = 0x00
                bytes[block * bytesPerBlock + off + 1] = 0x3C
            }
        }
        return bytes
    }

    private fun directStorage(bytes: ByteArray): DirectBufferStorage {
        val s = DirectBufferStorage.allocate(bytes.size)
        s.buffer().put(bytes)
        return s
    }

    @Test
    fun q4k_direct_buffer_decodes_like_heap() {
        val shape = Shape(4, 512) // 2 blocks per row, 8 blocks
        val bytes = randomPayload(8, 144, intArrayOf(0, 2), seed = 5)
        val heap = Q4_KBlockTensorData(shape, bytes)
        val buf = BufferPackedTensorData(shape, directStorage(bytes), TensorEncoding.Q4_K)

        assertContentEquals(heap.toFloatArray(), buf.toFloatArray())
        // TensorData.get mirrors the heap class's raw-code semantics (see class KDoc)
        assertEquals(heap.get(3, 511).toFloat(), buf.get(3, 511))
        assertContentEquals(heap.copyToFloatArray(), buf.copyToFloatArray())
        assertEquals(bytes.size.toLong(), buf.physicalBytes)
        assertEquals(BlockOrder.ROW_MAJOR, buf.blockOrder)
    }

    @Test
    fun q6k_mapped_file_decodes_like_heap() {
        val shape = Shape(2, 512)
        val bytes = randomPayload(4, 210, intArrayOf(208), seed = 9)
        val heap = Q6_KBlockTensorData(shape, bytes)

        val file = Files.createTempFile("bpt-q6k", ".bin")
        try {
            Files.write(file, bytes)
            val storage = MappedBufferStorage.map(file, 0, bytes.size.toLong())
            val buf = BufferPackedTensorData(shape, storage, TensorEncoding.Q6_K)
            assertContentEquals(heap.toFloatArray(), buf.toFloatArray())
            assertEquals(heap.get(1, 300).toFloat(), buf.get(1, 300))
        } finally {
            Files.deleteIfExists(file)
        }
    }

    @Test
    fun packed_view_is_row_major_over_the_same_storage() {
        val shape = Shape(2, 256)
        val bytes = randomPayload(2, 144, intArrayOf(0, 2), seed = 3)
        val storage = directStorage(bytes)
        val buf = BufferPackedTensorData(shape, storage, TensorEncoding.Q4_K)

        val view = buf.packedView
        assertEquals(BlockOrder.ROW_MAJOR, view.layout.blockOrder)
        assertTrue(view.storage === storage, "packedView must borrow the off-heap storage, not copy")
        // rule 4: view.get decodes — spot-check against the heap decode
        val heap = Q4_KBlockTensorData(shape, bytes)
        assertEquals(heap.toFloatArray()[1 * 256 + 17], view.get(1, 17))
    }

    @Test
    fun packedData_refuses_a_heap_copy() {
        val bytes = randomPayload(1, 144, intArrayOf(0, 2), seed = 1)
        val buf = BufferPackedTensorData(Shape(1, 256), directStorage(bytes), TensorEncoding.Q4_K)
        assertFailsWith<UnsupportedOperationException> { buf.packedData }
        assertFailsWith<UnsupportedOperationException> { buf.set(0, 0, value = 1f) }
    }

    @Test
    fun size_and_encoding_are_validated() {
        val bytes = randomPayload(2, 144, intArrayOf(0, 2), seed = 2)
        assertFailsWith<IllegalArgumentException> {
            // storage holds 2 blocks, shape wants 4
            BufferPackedTensorData(Shape(4, 256), directStorage(bytes), TensorEncoding.Q4_K)
        }
        assertFailsWith<IllegalArgumentException> {
            BufferPackedTensorData(Shape(2, 256), directStorage(bytes), TensorEncoding.Q8_0)
        }
    }
}
