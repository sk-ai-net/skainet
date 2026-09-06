package sk.ainet.lang.memory

import sk.ainet.lang.tensor.storage.MemoryDomain
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertIs

class PlatformStorageJvmTest {
    @Test
    fun jvmBindsSegmentsAndMappedFiles() {
        assertIs<SegmentStorage>(PlatformStorage.allocate(16, MemoryDomain.HOST_OFFHEAP))
        assertIs<Storage.Heap>(PlatformStorage.allocate(16, MemoryDomain.HOST_HEAP))
        val f = Files.createTempFile("skainet-ps", ".bin"); f.toFile().deleteOnExit(); Files.write(f, ByteArray(64) { it.toByte() })
        val m = PlatformStorage.mapFile(f.toString(), 8, 16)
        assertIs<MappedFileStorage>(m); assertEquals(16L, m.sizeBytes); assertEquals(8.toByte(), m.segment().get(java.lang.foreign.ValueLayout.JAVA_BYTE, 0))
        m.close()
        assertEquals("OffHeap=MemorySegment (FFM) · Mapped=FileChannel.map → MemorySegment", PlatformStorage.info.toString())
    }

    @Test
    fun copyIntoAndCopyFromRoundTripOnSegmentStorage() {
        val storage = PlatformStorage.allocate(8, MemoryDomain.HOST_OFFHEAP)
        val written = ByteArray(8) { (it + 1).toByte() }
        storage.copyFrom(written)
        val readBack = ByteArray(8)
        storage.copyInto(readBack)
        assertContentEquals(written, readBack)

        // Partial, offset copies agree too.
        val partial = ByteArray(3)
        storage.copyInto(partial, offset = 2, length = 3)
        assertContentEquals(byteArrayOf(3, 4, 5), partial)
        storage.close()
    }

    @Test
    fun copyIntoReadsMappedFileBytes() {
        val f = Files.createTempFile("skainet-ps-copy", ".bin"); f.toFile().deleteOnExit()
        Files.write(f, ByteArray(16) { it.toByte() })
        val mapped = PlatformStorage.mapFile(f.toString(), 4, 8)
        val out = ByteArray(8)
        mapped.copyInto(out)
        assertContentEquals(ByteArray(8) { (it + 4).toByte() }, out)
        mapped.close()
    }
}
