package sk.ainet.lang.memory

import sk.ainet.lang.tensor.TensorId
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** SKEEP-003 §4.7 / §8 item 4, PRD M1-F9: the debug mode that would have found #782 in one run. */
class MemoryDebugTest {

    @BeforeTest fun on() { MemoryDebug.overrideEnabled = true; MemoryDebug.reset() }

    @AfterTest fun off() {
        MemoryDebug.onAllocate = null; MemoryDebug.onClose = null; MemoryDebug.onAdapter = null; MemoryDebug.onLeak = null
        MemoryDebug.reset(); MemoryDebug.overrideEnabled = null
    }

    @Test
    fun offByDefaultAndCostsNothing() {
        MemoryDebug.overrideEnabled = null
        // the platform default is off unless the env/property says otherwise
        val before = MemoryDebug.liveEntries().size
        Storage.Heap.floats(4).close()
        assertEquals(before, MemoryDebug.liveEntries().size, "nothing is recorded while debug mode is off")
    }

    @Test
    fun allocationsAreTaggedWithScopeOriginAndSite() {
        val id = TensorId.parse("model.layers[3].mlp.down_proj.weight")
        var seen: MemoryDebug.Entry? = null
        MemoryDebug.onAllocate = { seen = it }
        val s = Storage.Heap.floats(64, ScopeKind.MODEL, origin = id)
        val e = assertNotNull(seen)
        assertEquals(s.id, e.storageId); assertEquals(ScopeKind.MODEL, e.scope); assertEquals(256L, e.bytes)
        assertEquals(id, e.origin); assertFalse(e.closed)
        assertEquals(e, MemoryDebug.entry(s.id))
        assertEquals(256L, MemoryDebug.liveBytes(ScopeKind.MODEL))
        assertEquals(256L, MemoryDebug.peakBytes(ScopeKind.MODEL))
        assertTrue(MemoryDebug.liveEntries().any { it.storageId == s.id })
        s.close()
        assertTrue(assertNotNull(MemoryDebug.entry(s.id)).closed)
        assertEquals(0L, MemoryDebug.liveBytes(ScopeKind.MODEL))
        assertEquals(256L, MemoryDebug.peakBytes(ScopeKind.MODEL), "the peak survives the free")
    }

    @Test
    fun useAfterCloseNamesTheStorageAndItsOrigin() {
        val id = TensorId.parse("model.embed_tokens.weight")
        val s = Storage.Heap.floats(8, ScopeKind.MODEL, origin = id)
        s.close()
        val ex = assertFailsWith<StorageClosedException> { s.checkAlive() }
        val msg = assertNotNull(ex.message)
        assertTrue(msg.contains("model.embed_tokens.weight"), msg)
        assertTrue(msg.contains("model scope"), msg)
        assertTrue(msg.contains("32 B"), msg)
    }

    @Test
    fun closeHookFiresOnce() {
        var closes = 0
        MemoryDebug.onClose = { closes++ }
        val s = Storage.Heap.bytes(16)
        s.close(); s.close()
        assertEquals(1, closes)
    }

    @Test
    fun adapterInsertionsAreReported() {
        var kind: String? = null; var bytes = 0L; var target: TensorId? = null
        MemoryDebug.onAdapter = { k, b, t -> kind = k; bytes = b; target = t }
        MemoryDebug.recordAdapter("dequantize", 96L * 1024 * 1024, TensorId.parse("model.layers[3].mlp.down_proj.weight"))
        assertEquals("dequantize", kind); assertEquals(96L * 1024 * 1024, bytes)
        assertEquals("model.layers[3].mlp.down_proj.weight", assertNotNull(target).canonical)
    }

    @Test
    fun leakReportingNamesTheStoragesThatOutlivedAReset() {
        var leaked: List<MemoryDebug.Entry> = emptyList()
        MemoryDebug.onLeak = { leaked = it }
        val escaped = Storage.Heap.floats(4, ScopeKind.FORWARD, origin = TensorId.parse("model.act#step=1"))
        val closed = Storage.Heap.floats(4, ScopeKind.FORWARD)
        closed.close()
        val reported = MemoryDebug.reportLeaks(listOf(escaped.id, closed.id))
        assertEquals(1, reported.size, "only the still-open storage is a leak")
        assertEquals(escaped.id, reported.single().storageId)
        assertEquals(reported, leaked)
        assertTrue(MemoryDebug.reportLeaks(emptyList()).isEmpty())
    }

    @Test
    fun theReportShowsPeaksAndLiveStorages() {
        Storage.Heap.floats(32, ScopeKind.MODEL, origin = TensorId.parse("model.w"))
        val tmp = Storage.Heap.floats(8, ScopeKind.FORWARD); tmp.close()
        val text = MemoryDebug.report()
        assertTrue(text.contains("storages seen")); assertTrue(text.contains("model"))
        assertTrue(text.contains("model.w"), text)
        assertTrue(text.contains("peak"))
        MemoryDebug.reset()
        assertEquals(0L, MemoryDebug.peakBytes(ScopeKind.MODEL))
        assertNull(MemoryDebug.entry(StorageId(1)))
    }
}
