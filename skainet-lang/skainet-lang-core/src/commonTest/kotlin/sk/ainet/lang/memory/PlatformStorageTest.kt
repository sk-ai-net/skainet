package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.storage.MemoryDomain
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** SKEEP-003 §4.8: one common door to the platform's storage kinds; heap everywhere, off-heap where the target has it. */
class PlatformStorageTest {

    @Test
    fun heapIsSupportedEverywhereAndAllocatesHeapStorage() {
        assertTrue(PlatformStorage.supports(MemoryDomain.HOST_HEAP))
        val sink = RecordingTraceSink()
        val s = PlatformStorage.allocate(64, MemoryDomain.HOST_HEAP, ScopeKind.FORWARD, sink = sink)
        assertIs<Storage.Heap>(s); assertEquals(64L, s.sizeBytes); assertEquals(ScopeKind.FORWARD, s.scope); assertEquals(MemoryDomain.HOST_HEAP, s.domain)
        assertEquals(64L, assertIs<TraceEvent.Allocation>(sink.events().single()).bytes)
        s.close(); assertFalse(s.isAlive)
    }

    @Test
    fun offHeapRequestYieldsOffHeapOrTheHeapFallback() {
        val s = PlatformStorage.allocate(128, MemoryDomain.HOST_OFFHEAP, ScopeKind.MODEL)
        if (PlatformStorage.supports(MemoryDomain.HOST_OFFHEAP)) {
            assertIs<Storage.OffHeap>(s); assertEquals(MemoryDomain.HOST_OFFHEAP, s.domain)
        } else {
            assertIs<Storage.Heap>(s); assertEquals(MemoryDomain.HOST_HEAP, s.domain) // JS/Wasm
        }
        assertEquals(128L, s.sizeBytes); assertEquals(ScopeKind.MODEL, s.scope); assertTrue(s.isMutable)
        val v = s.slice(32, 64); assertEquals(64L, v.sizeBytes); assertIs<Owner.Alias>(v.owner)
        s.close(); assertFalse(v.isAlive)
        assertFailsWith<StorageClosedException> { s.checkAlive() }
    }

    @Test
    fun deviceAndFileDomainsAreNotAllocatable() {
        assertFailsWith<IllegalArgumentException> { PlatformStorage.allocate(1, MemoryDomain.DEVICE_LOCAL) }
        assertFailsWith<IllegalArgumentException> { PlatformStorage.allocate(1, MemoryDomain.MMAP_FILE) }
        assertFalse(PlatformStorage.supports(MemoryDomain.DEVICE_LOCAL))
    }

    @Test
    fun mappedFilesAreEitherSupportedOrClearlyUnsupported() {
        if (!PlatformStorage.supportsMappedFiles) {
            assertFailsWith<UnsupportedOperationException> { PlatformStorage.mapFile("/nonexistent", 0, 1) }
        } else {
            assertTrue(PlatformStorage.supports(MemoryDomain.MMAP_FILE))
        }
    }
}
