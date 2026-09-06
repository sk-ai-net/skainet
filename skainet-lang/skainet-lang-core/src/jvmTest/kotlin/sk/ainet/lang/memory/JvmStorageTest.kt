package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** SKEEP-003 §4.8.1/§4.8.2: OffHeap (MemorySegment / direct ByteBuffer) and Mapped (FileChannel.map) storage kinds. */
class JvmStorageTest {

    @Test
    fun segmentStorageOwnsItsArenaAndFreesOnClose() {
        val sink = RecordingTraceSink()
        val id = TensorId.parse("model.layers[0].attn.q#step=3")
        val s = SegmentStorage.allocate(bytes = 1024, scope = ScopeKind.FORWARD, origin = id, sink = sink)
        assertEquals(MemoryDomain.HOST_OFFHEAP, s.domain); assertEquals(ScopeKind.FORWARD, s.scope); assertTrue(s.isMutable)
        assertEquals(1024L, s.sizeBytes); assertEquals(0L, s.segment().address() % 64)
        s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 3, 1.5f)
        assertEquals(1.5f, s.segment().getAtIndex(ValueLayout.JAVA_FLOAT, 3))
        val alloc = assertIs<TraceEvent.Allocation>(sink.events().single()); assertEquals(1024L, alloc.bytes); assertEquals(id, alloc.origin)
        s.close()
        assertFalse(s.isAlive)
        assertFailsWith<StorageClosedException> { s.segment() }
        assertIs<TraceEvent.Free>(sink.events()[1])
    }

    @Test
    fun segmentStorageInACallerArenaIsFreedByTheArenaNotByClose() {
        Arena.ofShared().use { arena ->
            val s = SegmentStorage.allocate(256, ScopeKind.MODEL, arena = arena)
            val seg = s.segment()
            s.close()               // marks dead, does not close the caller's arena
            assertTrue(seg.scope().isAlive)
            assertFailsWith<StorageClosedException> { s.segment() }
        }
    }

    @Test
    fun segmentSliceIsAnAliasOverTheSameBytes() {
        val s = SegmentStorage.allocate(64)
        val v = s.slice(16, 32)
        assertIs<Owner.Alias>(v.owner); assertEquals(32L, v.sizeBytes)
        v.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 0, 9f)
        assertEquals(9f, s.segment().getAtIndex(ValueLayout.JAVA_FLOAT, 4))
        assertFailsWith<IllegalArgumentException> { s.slice(60, 8) }
        s.close(); assertFalse(v.isAlive)
    }

    @Test
    fun borrowedSegmentAndArrayAreNeverFreed() {
        val arr = FloatArray(8) { it.toFloat() }
        val s = SegmentStorage.borrow(arr)
        assertIs<Owner.Borrowed>(s.owner); assertEquals(32L, s.sizeBytes); assertTrue(s.isMutable)
        s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, 1, 42f)
        assertEquals(42f, arr[1]) // zero-copy over the caller's array
        s.close()
        assertEquals(42f, arr[1])
        val ro = SegmentStorage.borrow(MemorySegment.ofArray(IntArray(2)).asReadOnly())
        assertFalse(ro.isMutable)
    }

    @Test
    fun mappedFileStorageReadsTheFileAndUnmapsOnClose() {
        val f = Files.createTempFile("skainet-mapped", ".bin"); f.toFile().deleteOnExit()
        val bytes = ByteArray(256) { it.toByte() }; Files.write(f, bytes)
        val sink = RecordingTraceSink()
        val m = MappedFileStorage.map(f, fileOffset = 16, length = 64, origin = TensorId.parse("model.w"), sink = sink)
        assertEquals(MemoryDomain.MMAP_FILE, m.domain); assertEquals(ScopeKind.MODEL, m.scope); assertFalse(m.isMutable)
        assertEquals(64L, m.sizeBytes); assertEquals(16L, m.fileOffset)
        assertEquals(16.toByte(), m.segment().get(ValueLayout.JAVA_BYTE, 0)); assertEquals(79.toByte(), m.segment().get(ValueLayout.JAVA_BYTE, 63))
        val v = m.slice(8, 8); assertEquals(24L, v.fileOffset); assertEquals(24.toByte(), v.segment().get(ValueLayout.JAVA_BYTE, 0))
        assertTrue(m.toString().startsWith("Mapped(#")); assertTrue(m.toString().contains("@0x10"))
        assertEquals(f.toString(), assertIs<TraceEvent.Allocation>(sink.events().single()).site)
        m.close()
        assertFalse(m.isAlive); assertFalse(v.isAlive)
        assertFailsWith<StorageClosedException> { m.segment() }
    }

    /**
     * `TensorView.get()` over dense FP32 backed by [SegmentStorage] / [MappedFileStorage] — was
     * `UnsupportedOperationException("element access over SegmentStorage needs a platform reader
     * (use a kernel)")` (the reference-kernel fallback assumed every non-Heap storage had a real
     * kernel to serve it; a MAPPED weight with no matching registered kernel key had none). Found
     * via a real Gemma 4 GGUF's `per_layer_model_proj.weight` (a MAPPED dense weight, PLE) falling
     * to `ReferenceMatmulKernel` and throwing instead of just being slow.
     */
    @Test
    fun denseFp32ViewReadsThroughSegmentStorage() {
        val s = SegmentStorage.allocate(bytes = 4L * 6)
        for (i in 0 until 6) s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, i.toLong(), i.toFloat() + 0.5f)
        val v = TensorView.dense(s, Shape(2, 3), FP32, TensorId.parse("model.ple_proj"))
        assertEquals(0.5f, v.get(0, 0)); assertEquals(3.5f, v.get(1, 0)); assertEquals(5.5f, v.get(1, 2))
        assertContentEquals(floatArrayOf(0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f), v.toFloatArray())
        s.close()
    }

    @Test
    fun denseFp32ViewReadsThroughMappedFileStorage() {
        val f = Files.createTempFile("skainet-mapped-dense", ".bin"); f.toFile().deleteOnExit()
        val bytes = ByteArray(4 * 4)
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until 4) bb.putFloat(i * 4, i.toFloat() * 10f)
        Files.write(f, bytes)
        val m = MappedFileStorage.map(f, fileOffset = 0, length = 16, origin = TensorId.parse("model.w"))
        val v = TensorView.dense(m, Shape(4), FP32)
        assertEquals(0f, v.get(0)); assertEquals(10f, v.get(1)); assertEquals(30f, v.get(3))
        m.close()
    }

    /**
     * Same gap, the narrow-float (BF16/FP16) decode path: `NarrowFloatDecoder.decodeAt` threw
     * "narrow-float views need heap storage in this milestone" for anything but [Storage.Heap].
     */
    @Test
    fun narrowFloatViewReadsThroughSegmentStorage() {
        val codec = sk.ainet.lang.types.Bf16Codec
        val s = SegmentStorage.allocate(bytes = 2L * 3)
        val values = floatArrayOf(1.0f, -2.5f, 100.0f)
        for (i in values.indices) {
            val bits = codec.encode(values[i])
            s.segment().set(ValueLayout.JAVA_BYTE, (i * 2).toLong(), (bits and 0xFF).toByte())
            s.segment().set(ValueLayout.JAVA_BYTE, (i * 2 + 1).toLong(), ((bits ushr 8) and 0xFF).toByte())
        }
        val shape = Shape(3)
        val view = TensorView(
            shape = shape,
            format = Format(codec.dtype, sk.ainet.lang.tensor.storage.TensorEncoding.Dense(2)),
            layout = Layout(shape = shape, strides = Layout.rowMajorStrides(shape), elementBytes = 2),
            storage = s,
            decoder = NarrowFloatDecoder(codec),
        )
        for (i in values.indices) {
            assertTrue(kotlin.math.abs(view.get(i) - values[i]) <= kotlin.math.abs(values[i]) * 0.01f + 0.01f)
        }
        s.close()
    }

    @Test
    fun directBufferStorageAllocatesBorrowsAndSlices() {
        val sink = RecordingTraceSink()
        val d = DirectBufferStorage.allocate(64, ScopeKind.FORWARD, sink = sink)
        assertEquals(MemoryDomain.HOST_OFFHEAP, d.domain); assertTrue(d.isMutable); assertTrue(d.buffer().isDirect)
        assertEquals(ByteOrder.LITTLE_ENDIAN, d.buffer().order())
        d.buffer().putFloat(8, 2.5f)
        assertEquals(2.5f, d.buffer().getFloat(8))
        val v = d.slice(8, 8); assertEquals(2.5f, v.buffer().getFloat(0)); assertIs<Owner.Alias>(v.owner)
        assertEquals(64L, assertIs<TraceEvent.Allocation>(sink.events().single()).bytes)
        val b = DirectBufferStorage.borrow(ByteBuffer.allocate(16).asReadOnlyBuffer())
        assertFalse(b.isMutable); assertIs<Owner.Borrowed>(b.owner)
        d.close(); assertFalse(v.isAlive)
        assertFailsWith<StorageClosedException> { d.buffer() }
    }

    @Test
    fun mappedBufferStorageReadsTheFile() {
        val f = Files.createTempFile("skainet-mapped-bb", ".bin"); f.toFile().deleteOnExit()
        Files.write(f, ByteArray(128) { (it * 2).toByte() })
        val m = MappedBufferStorage.map(f, 32, 32)
        assertEquals(64.toByte(), m.buffer().get(0)); assertEquals(32L, m.sizeBytes); assertFalse(m.isMutable)
        assertEquals(ScopeKind.MODEL, m.scope)
        val v = m.slice(4, 4); assertEquals(36L, v.fileOffset); assertEquals(72.toByte(), v.buffer().get(0))
        m.close(); assertFailsWith<StorageClosedException> { m.buffer() }
    }

    @Test
    fun allKindsAreStoragesWithDistinctIds() {
        val h = Storage.Heap.floats(1); val s = SegmentStorage.allocate(4); val d = DirectBufferStorage.allocate(4)
        val ids = listOf(h.id.value, s.id.value, d.id.value)
        assertEquals(3, ids.toSet().size)
        assertEquals(Storage.Heap::class, h::class)
        assertTrue(s is Storage.OffHeap && d is Storage.OffHeap)
        listOf<Storage>(h, s, d).forEach { it.close() }
    }
}
