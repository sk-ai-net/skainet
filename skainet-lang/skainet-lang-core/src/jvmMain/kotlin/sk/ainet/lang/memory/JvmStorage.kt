package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * JVM binding of [Storage.OffHeap]: a `MemorySegment` (FFM) that is not scanned or copied by the GC,
 * may exceed 2 GB, is handed to IREE/JNI zero-copy, and is freed deterministically by the `Arena`
 * that owns it (SKEEP-003 §4.8.1). An owned storage allocated without an explicit arena gets its
 * own `Arena.ofShared()` and closes it on [close]; milestone slice #1021 (`Scope`) passes the
 * scope's arena instead, so `Forward` becomes a recycled bump slab.
 */
public class SegmentStorage private constructor(
    override val id: StorageId,
    private val seg: MemorySegment,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
    private val ownedArena: Arena?,
    private val mutable: Boolean,
) : Storage.OffHeap() {
    override val sizeBytes: Long get() = seg.byteSize()
    override val isMutable: Boolean get() = (owner as? Owner.Alias)?.parent?.isMutable ?: mutable

    /** The segment — kernels take it once per call (`ByteVector.fromMemorySegment`, `getAtIndex`). Throws when closed. */
    public fun segment(): MemorySegment { checkAlive(); return seg }

    override fun slice(offsetBytes: Long, lengthBytes: Long): SegmentStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        return SegmentStorage(StorageId.next(), seg.asSlice(offsetBytes, lengthBytes), Owner.Alias(this), debugOrigin, sink, null, mutable)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, dest, destOffset, length)
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        MemorySegment.copy(src, srcOffset, seg, ValueLayout.JAVA_BYTE, offset, length)
    }

    override fun onClose() { ownedArena?.close() }

    public companion object {
        /**
         * Allocate [bytes] zeroed off-heap bytes (aligned to [alignment]) owned by a scope of kind [scope].
         * With [arena] `null` the storage owns a private `Arena.ofShared()`; pass the scope's arena to
         * let the scope free it (the `Forward` slab pattern).
         */
        public fun allocate(bytes: Long, scope: ScopeKind = ScopeKind.AMBIENT, arena: Arena? = null, alignment: Long = 64L, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): SegmentStorage {
            require(bytes >= 0) { "bytes must be >= 0" }
            val owned = arena ?: Arena.ofShared()
            val seg = owned.allocate(bytes, alignment)
            val s = SegmentStorage(StorageId.next(), seg, Owner.Owned(scope), origin, sink, if (arena == null) owned else null, true)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, bytes, origin))
            return s
        }

        /** Borrow an existing segment (a loader's, IREE's, a caller's `MemorySegment.ofArray`) — never freed by us. */
        public fun borrow(segment: MemorySegment, mutable: Boolean = !segment.isReadOnly, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): SegmentStorage =
            SegmentStorage(StorageId.next(), segment, Owner.Borrowed(segment), origin, sink, null, mutable)

        /** Borrow a heap array zero-copy as a segment (`MemorySegment.ofArray`). */
        public fun borrow(array: FloatArray, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): SegmentStorage =
            borrow(MemorySegment.ofArray(array), mutable = true, origin = origin, sink = sink)

        /** Layout helper for float element access on a segment. */
        public val FLOAT: ValueLayout.OfFloat = ValueLayout.JAVA_FLOAT
    }
}

/**
 * JVM binding of [Storage.Mapped]: a read-only region of a file mapped with `FileChannel.map` into
 * an `Arena.ofShared()`. The OS pages the bytes; resident set = pages touched; the page cache is
 * shared across processes; closing the storage unmaps (SKEEP-003 §4.8.1). Packed GGUF weights and
 * embedding tables live here.
 */
public class MappedFileStorage private constructor(
    override val id: StorageId,
    public val path: Path,
    public val fileOffset: Long,
    private val seg: MemorySegment,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
    private val arena: Arena?,
) : Storage.Mapped() {
    override val sizeBytes: Long get() = seg.byteSize()
    override val isMutable: Boolean get() = false

    public fun segment(): MemorySegment { checkAlive(); return seg }

    override fun slice(offsetBytes: Long, lengthBytes: Long): MappedFileStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        return MappedFileStorage(StorageId.next(), path, fileOffset + offsetBytes, seg.asSlice(offsetBytes, lengthBytes), Owner.Alias(this), debugOrigin, sink, null)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, offset, dest, destOffset, length)
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        MemorySegment.copy(src, srcOffset, seg, ValueLayout.JAVA_BYTE, offset, length)
    }

    override fun onClose() { arena?.close() }

    override fun toString(): String = "Mapped(${id}, ${sizeBytes} B, $path @0x${fileOffset.toString(16)}${debugOrigin?.let { ", $it" } ?: ""}${if (isAlive) "" else ", closed"})"

    public companion object {
        /**
         * Map `[fileOffset, fileOffset + length)` of [path] read-only. Owned by a scope of kind [scope]
         * (normally `MODEL`): closing the storage — or, from #1021, the scope — unmaps.
         */
        public fun map(path: Path, fileOffset: Long, length: Long, scope: ScopeKind = ScopeKind.MODEL, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): MappedFileStorage {
            require(fileOffset >= 0 && length >= 0) { "offset/length must be >= 0" }
            val arena = Arena.ofShared()
            val seg = FileChannel.open(path, StandardOpenOption.READ).use { ch -> ch.map(FileChannel.MapMode.READ_ONLY, fileOffset, length, arena) }
            val s = MappedFileStorage(StorageId.next(), path, fileOffset, seg, Owner.Owned(scope), origin, sink, arena)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, length, origin, site = path.toString()))
            return s
        }
    }
}
