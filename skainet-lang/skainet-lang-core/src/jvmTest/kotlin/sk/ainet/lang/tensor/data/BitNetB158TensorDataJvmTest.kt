package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.PlatformStorage
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.MemoryDomain
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertIs

/**
 * #1202: an off-heap-backed [BitNetB158TensorData] must decode bit-for-bit identically to the
 * historical heap-backed one, and [BitNetB158TensorData.set] must write through to the backing
 * storage rather than only updating the lazily materialized snapshot.
 */
class BitNetB158TensorDataJvmTest {

    private fun ternaryValues(count: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(count) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1) * 0.5f
        }
    }

    @Test
    fun offHeapStorageDecodesIdenticallyToHeap() {
        val n = 6; val k = 32
        val values = ternaryValues(n * k, seed = 5)
        val bytes = TernaryCodec.encodeBitNet(values)
        val heap = BitNetB158TensorData(Shape(n, k), bytes)

        val storage = PlatformStorage.allocate(bytes.size.toLong(), MemoryDomain.HOST_OFFHEAP)
        storage.copyFrom(bytes)
        val offHeap = BitNetB158TensorData.fromStorage(Shape(n, k), storage)

        assertEquals(heap.scale, offHeap.scale)
        assertContentEquals(heap.toFloatArray(), offHeap.toFloatArray())
        assertContentEquals(heap.packedData, offHeap.packedData)
        assertEquals(heap.get(1, 1), offHeap.get(1, 1))
        assertIs<sk.ainet.lang.memory.Storage.OffHeap>(offHeap.packedStorage)
        storage.close()
    }

    @Test
    fun setWritesThroughToBackingStorage() {
        val values = floatArrayOf(1f, -1f, 0f, 1f, -1f, 0f, 0f, 1f)
        val bytes = TernaryCodec.encodeBitNet(values)
        val storage = PlatformStorage.allocate(bytes.size.toLong(), MemoryDomain.HOST_OFFHEAP)
        storage.copyFrom(bytes)
        val offHeap = BitNetB158TensorData.fromStorage(Shape(8), storage)

        offHeap.set(2, value = -1)
        assertEquals(-1, offHeap.get(2).toInt())

        // A second view constructed straight from the same storage — not the same Kotlin object,
        // no shared lazy snapshot — must see the write, proving it reached the storage itself.
        val reread = BitNetB158TensorData.fromStorage(Shape(8), storage)
        assertEquals(-1, reread.get(2).toInt())
        storage.close()
    }
}
