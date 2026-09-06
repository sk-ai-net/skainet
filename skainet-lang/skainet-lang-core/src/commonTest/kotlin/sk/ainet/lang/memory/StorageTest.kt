package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-003 rules 1–2 (one byte owner, ownership enforced) for the common Heap storage. */
class StorageTest {

    @Test
    fun ownedHeapStorageHasIdSizeScopeAndIsFreedOnce() {
        val sink = RecordingTraceSink()
        val id = TensorId.parse("model.layers[0].attn.scores#step=1")
        val s = Storage.Heap.floats(256, ScopeKind.FORWARD, origin = id, sink = sink)
        assertTrue(s.isAlive); assertTrue(s.isMutable)
        assertEquals(1024L, s.sizeBytes); assertEquals(256, s.elementCount); assertEquals(4, s.elementBytes)
        assertEquals(ScopeKind.FORWARD, s.scope); assertEquals(MemoryDomain.HOST_HEAP, s.domain)
        assertIs<Owner.Owned>(s.owner); assertEquals(id, s.debugOrigin)
        assertNotNull(s.floats); assertNull(s.ints); assertNull(s.bytes); assertEquals(0, s.arrayOffset)
        val alloc = assertIs<TraceEvent.Allocation>(sink.events().single())
        assertEquals(s.id.value, alloc.storageId); assertEquals(ScopeKind.FORWARD, alloc.scope); assertEquals(1024L, alloc.bytes); assertEquals(id, alloc.origin)

        s.close(); s.close() // idempotent
        assertFalse(s.isAlive)
        val free = assertIs<TraceEvent.Free>(sink.events()[1]); assertEquals(s.id.value, free.storageId); assertEquals(1024L, free.bytes)
        assertEquals(2, sink.events().size)
        val ex = assertFailsWith<StorageClosedException> { s.checkAlive() }
        assertEquals(s.id, ex.storageId); assertEquals(id, ex.origin); assertTrue(ex.message!!.contains("attn.scores"))
    }

    @Test
    fun storageIdsAreMonotonicAndDistinct() {
        val a = Storage.Heap.bytes(4); val b = Storage.Heap.ints(4); val c = Storage.Heap.floats(4)
        assertTrue(a.id.value < b.id.value && b.id.value < c.id.value)
        assertEquals("#${a.id.value}", a.id.toString())
        assertEquals(4L, a.sizeBytes); assertEquals(16L, b.sizeBytes); assertEquals(1, a.elementBytes)
    }

    @Test
    fun borrowedStorageIsNeverFreedOnlyReleased() {
        val sink = RecordingTraceSink()
        val arr = FloatArray(8) { it.toFloat() }
        val s = Storage.Heap.wrap(arr, offset = 2, count = 4, mutable = false, sink = sink)
        assertIs<Owner.Borrowed>(s.owner); assertSame(arr, (s.owner as Owner.Borrowed).external)
        assertEquals(ScopeKind.AMBIENT, s.scope); assertFalse(s.isMutable)
        assertSame(arr, s.floats); assertEquals(2, s.arrayOffset); assertEquals(16L, s.sizeBytes)
        assertTrue(sink.events().isEmpty()) // borrowing is not an allocation
        s.close() // release: forget, do not touch the caller's bytes
        assertFalse(s.isAlive)
        assertEquals(listOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f), arr.toList())
        assertFailsWith<StorageClosedException> { s.checkAlive() }
    }

    @Test
    fun aliasKeepsParentAliveDelegatesMutabilityAndDiesWithIt() {
        val parent = Storage.Heap.floats(16, ScopeKind.MODEL)
        val view = parent.slice(offsetBytes = 16, lengthBytes = 32)
        assertIs<Owner.Alias>(view.owner); assertSame(parent, (view.owner as Owner.Alias).parent)
        assertEquals(ScopeKind.MODEL, view.scope) // inherited
        assertSame(parent.floats, view.floats); assertEquals(4, view.arrayOffset); assertEquals(32L, view.sizeBytes); assertEquals(8, view.elementCount)
        assertTrue(view.isMutable)
        val roView = Storage.Heap.wrap(FloatArray(4), mutable = false).slice(0, 8)
        assertFalse(roView.isMutable) // delegated
        // nested alias
        val inner = view.slice(8, 8)
        assertEquals(6, inner.arrayOffset); assertEquals(8L, inner.sizeBytes)
        // closing an alias detaches it only
        inner.close(); assertFalse(inner.isAlive); assertTrue(view.isAlive); assertTrue(parent.isAlive)
        // closing the parent invalidates every alias
        parent.close()
        assertFalse(view.isAlive)
        assertFailsWith<StorageClosedException> { view.checkAlive() }
        assertFailsWith<StorageClosedException> { parent.slice(0, 4) }
    }

    @Test
    fun sliceBoundsAndAlignmentAreChecked() {
        val s = Storage.Heap.ints(4)
        assertFailsWith<IllegalArgumentException> { s.slice(0, 20) }
        assertFailsWith<IllegalArgumentException> { s.slice(-4, 4) }
        assertFailsWith<IllegalArgumentException> { s.slice(2, 4) } // not 4-byte aligned
        assertEquals(0L, s.slice(16, 0).sizeBytes)
        assertEquals(2, Storage.Heap.bytes(8).slice(2, 3).arrayOffset)
    }

    @Test
    fun toStringNamesKindIdSizeOwnerAndState() {
        val s = Storage.Heap.floats(2, origin = TensorId.parse("x.w"))
        val t = s.toString()
        assertTrue(t.startsWith("Heap(#"), t); assertTrue(t.contains("8 B")); assertTrue(t.contains("Owned(scope=AMBIENT)")); assertTrue(t.contains("x.w"))
        s.close(); assertTrue(s.toString().endsWith("closed)"))
    }
}
