@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlinx.cinterop.UnsafeNumber::class)

package sk.ainet.lang.memory

import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ByteVar
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import platform.posix.MAP_FAILED
import platform.posix.MAP_PRIVATE
import platform.posix.O_RDONLY
import platform.posix.PROT_READ
import platform.posix.close
import platform.posix.free
import platform.posix.malloc
import platform.posix.memcpy
import platform.posix.memset
import platform.posix.mmap
import platform.posix.munmap
import platform.posix.open

/**
 * Kotlin/Native binding of [Storage.OffHeap]: `malloc`ed bytes (zeroed), freed on [close]
 * (SKEEP-003 §4.8.3). Alignment: `malloc` gives 16 bytes on every supported platform, which is what
 * NEON wants; 64-byte alignment for AMX-class paths is a follow-up with `posix_memalign`.
 */
public class NativeMallocStorage private constructor(
    override val id: StorageId,
    private val ptr: CPointer<ByteVar>,
    override val sizeBytes: Long,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
    private val mutable: Boolean,
) : Storage.OffHeap() {
    override val isMutable: Boolean get() = (owner as? Owner.Alias)?.parent?.isMutable ?: mutable

    /** Raw pointer + byte offset is what a C kernel receives; throws after close. */
    public fun pointer(): CPointer<ByteVar> { checkAlive(); return ptr }

    override fun slice(offsetBytes: Long, lengthBytes: Long): NativeMallocStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        return NativeMallocStorage(StorageId.next(), (ptr.rawValue + offsetBytes).toLong().toCPointer(), lengthBytes, Owner.Alias(this), debugOrigin, sink, mutable)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        val src = (ptr.rawValue + offset).toLong().toCPointer()
        dest.usePinned { pinned -> memcpy(pinned.addressOf(destOffset), src, length.convert()) }
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        val dst = (ptr.rawValue + offset).toLong().toCPointer()
        src.usePinned { pinned -> memcpy(dst, pinned.addressOf(srcOffset), length.convert()) }
    }

    override fun onClose() { if (owner is Owner.Owned) free(ptr) }

    public companion object {
        public fun allocate(bytes: Long, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): NativeMallocStorage {
            require(bytes >= 0) { "bytes must be >= 0" }
            val p = malloc(maxOf(bytes, 1L).convert()) ?: throw IllegalStateException("malloc($bytes) failed")
            memset(p, 0, maxOf(bytes, 1L).convert())
            val s = NativeMallocStorage(StorageId.next(), p.reinterpret(), bytes, Owner.Owned(scope), origin, sink, true)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, bytes, origin))
            return s
        }

        /** Borrow a caller's buffer — never freed by us. */
        public fun borrow(pointer: CPointer<ByteVar>, bytes: Long, mutable: Boolean = true, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): NativeMallocStorage =
            NativeMallocStorage(StorageId.next(), pointer, bytes, Owner.Borrowed(pointer), origin, sink, mutable)
    }
}

/** Kotlin/Native binding of [Storage.Mapped]: `mmap(2)` of a file region, read-only, `munmap` on close. */
public class NativeMappedStorage private constructor(
    override val id: StorageId,
    public val path: String,
    public val fileOffset: Long,
    private val base: COpaquePointer,
    private val ptr: CPointer<ByteVar>,
    override val sizeBytes: Long,
    private val mappedLength: Long,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
) : Storage.Mapped() {
    override val isMutable: Boolean get() = false

    public fun pointer(): CPointer<ByteVar> { checkAlive(); return ptr }

    override fun slice(offsetBytes: Long, lengthBytes: Long): NativeMappedStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
        return NativeMappedStorage(StorageId.next(), path, fileOffset + offsetBytes, base, (ptr.rawValue + offsetBytes).toLong().toCPointer(), lengthBytes, 0L, Owner.Alias(this), debugOrigin, sink)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        val src = (ptr.rawValue + offset).toLong().toCPointer()
        dest.usePinned { pinned -> memcpy(pinned.addressOf(destOffset), src, length.convert()) }
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        val dst = (ptr.rawValue + offset).toLong().toCPointer()
        src.usePinned { pinned -> memcpy(dst, pinned.addressOf(srcOffset), length.convert()) }
    }

    override fun onClose() { if (owner is Owner.Owned) munmap(base, mappedLength.convert()) }

    public companion object {
        private const val PAGE: Long = 4096L

        public fun map(path: String, fileOffset: Long, length: Long, scope: ScopeKind = ScopeKind.MODEL, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): NativeMappedStorage {
            require(fileOffset >= 0 && length >= 0) { "offset/length must be >= 0" }
            val fd = open(path, O_RDONLY)
            require(fd >= 0) { "cannot open $path" }
            try {
                val pageStart = fileOffset - (fileOffset % PAGE)
                val delta = fileOffset - pageStart
                val mapLen = maxOf(length + delta, 1L)
                val p = mmap(null, mapLen.convert(), PROT_READ, MAP_PRIVATE, fd, pageStart.convert())
                require(p != null && p != MAP_FAILED) { "mmap($path, $fileOffset, $length) failed" }
                val s = NativeMappedStorage(StorageId.next(), path, fileOffset, p, (p.rawValue + delta).toLong().toCPointer(), length, mapLen, Owner.Owned(scope), origin, sink)
                if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, length, origin, site = path))
                return s
            } finally {
                close(fd)
            }
        }
    }
}

private fun Long.toCPointer(): CPointer<ByteVar> = kotlinx.cinterop.interpretCPointer<ByteVar>(kotlinx.cinterop.nativeNullPtr + this)
    ?: throw IllegalStateException("null pointer")
