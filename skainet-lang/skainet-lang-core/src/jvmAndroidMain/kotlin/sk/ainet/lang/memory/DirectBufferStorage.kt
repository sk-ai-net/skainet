package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * Android (and JVM fallback) binding of [Storage.OffHeap]: a direct `ByteBuffer` — outside the ART
 * heap and its per-app limit (the root of #922), counted by the OS against the process. Freed when
 * the buffer becomes unreachable (direct buffers have no explicit free on Android); [close] marks
 * the storage dead so no late access can see it. Use [SegmentStorage] where FFM is available.
 */
public class DirectBufferStorage private constructor(
    override val id: StorageId,
    private val buf: ByteBuffer,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
) : Storage.OffHeap() {
    override val sizeBytes: Long get() = buf.capacity().toLong()
    override val isMutable: Boolean get() = !buf.isReadOnly

    /** An independent little-endian duplicate of the buffer (position 0); kernels take it once per call. */
    public fun buffer(): ByteBuffer { checkAlive(); return buf.duplicate().order(ByteOrder.LITTLE_ENDIAN) }

    override fun slice(offsetBytes: Long, lengthBytes: Long): DirectBufferStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        // Java 8 signatures on Android: Buffer.position/limit return Buffer, so keep the ByteBuffer typed.
        val d: ByteBuffer = buf.duplicate()
        d.position(offsetBytes.toInt()); d.limit((offsetBytes + lengthBytes).toInt())
        val s: ByteBuffer = d.slice().order(ByteOrder.LITTLE_ENDIAN)
        return DirectBufferStorage(StorageId.next(), s, Owner.Alias(this), debugOrigin, sink)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        val d: ByteBuffer = buf.duplicate()
        d.position(offset.toInt())
        d.get(dest, destOffset, length)
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        val d: ByteBuffer = buf.duplicate()
        d.position(offset.toInt())
        d.put(src, srcOffset, length)
    }

    public companion object {
        /** Allocate [bytes] zeroed direct bytes owned by a scope of kind [scope]. */
        public fun allocate(bytes: Int, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): DirectBufferStorage {
            require(bytes >= 0) { "bytes must be >= 0" }
            val s = DirectBufferStorage(StorageId.next(), ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN), Owner.Owned(scope), origin, sink)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, bytes.toLong(), origin))
            return s
        }

        /** Borrow a caller's buffer (direct or mapped) — never freed by us. */
        public fun borrow(buffer: ByteBuffer, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): DirectBufferStorage =
            DirectBufferStorage(StorageId.next(), buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN), Owner.Borrowed(buffer), origin, sink)
    }
}

/**
 * Android (and JVM fallback) binding of [Storage.Mapped]: a `MappedByteBuffer` from `FileChannel.map`
 * — weights outside ART entirely (SKEEP-002 / #921). Unmapped when the buffer becomes unreachable;
 * [close] marks the storage dead.
 */
public class MappedBufferStorage private constructor(
    override val id: StorageId,
    public val path: Path,
    public val fileOffset: Long,
    private val buf: ByteBuffer,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
) : Storage.Mapped() {
    override val sizeBytes: Long get() = buf.capacity().toLong()
    override val isMutable: Boolean get() = false

    public fun buffer(): ByteBuffer { checkAlive(); return buf.duplicate().order(ByteOrder.LITTLE_ENDIAN) }

    override fun slice(offsetBytes: Long, lengthBytes: Long): MappedBufferStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        val d: ByteBuffer = buf.duplicate()
        d.position(offsetBytes.toInt()); d.limit((offsetBytes + lengthBytes).toInt())
        val s: ByteBuffer = d.slice().order(ByteOrder.LITTLE_ENDIAN)
        return MappedBufferStorage(StorageId.next(), path, fileOffset + offsetBytes, s, Owner.Alias(this), debugOrigin, sink)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        val d: ByteBuffer = buf.duplicate()
        d.position(offset.toInt())
        d.get(dest, destOffset, length)
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        val d: ByteBuffer = buf.duplicate()
        d.position(offset.toInt())
        d.put(src, srcOffset, length)
    }

    public companion object {
        /** Map `[fileOffset, fileOffset + length)` of [path] read-only (length ≤ 2 GB per buffer). */
        public fun map(path: Path, fileOffset: Long, length: Long, scope: ScopeKind = ScopeKind.MODEL, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): MappedBufferStorage {
            require(fileOffset >= 0 && length in 0..Int.MAX_VALUE.toLong()) { "offset must be >= 0 and length in [0, 2 GB)" }
            val mbb: MappedByteBuffer = FileChannel.open(path, StandardOpenOption.READ).use { ch -> ch.map(FileChannel.MapMode.READ_ONLY, fileOffset, length) }
            val s = MappedBufferStorage(StorageId.next(), path, fileOffset, mbb.order(ByteOrder.LITTLE_ENDIAN), Owner.Owned(scope), origin, sink)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, length, origin, site = path.toString()))
            return s
        }
    }
}
