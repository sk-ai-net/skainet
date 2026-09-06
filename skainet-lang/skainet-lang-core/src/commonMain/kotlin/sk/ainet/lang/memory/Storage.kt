@file:OptIn(kotlin.concurrent.atomics.ExperimentalAtomicApi::class)

package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import kotlin.concurrent.atomics.AtomicLong
import kotlin.concurrent.atomics.fetchAndIncrement
import kotlin.jvm.JvmInline

/**
 * Monotonic per-process identity of one allocation (SKEEP-003 §0 *StorageId*): what the memory
 * debugger keys on. One `TensorId` maps to many storage ids over time (a `Forward` scope is
 * recycled every step); one storage may back many `TensorId`s (views, KV ring).
 */
@JvmInline
public value class StorageId(public val value: Long) {
    override fun toString(): String = "#$value"

    public companion object {
        private val counter = AtomicLong(1L)
        /** The next id; thread-safe. */
        public fun next(): StorageId = StorageId(counter.fetchAndIncrement())
    }
}

/**
 * How a [Storage] came to hold its bytes (SKEEP-003 §0 *Owner*, §4.4). Ownership is a
 * constructor argument, then it is enforced: [Borrowed] storage cannot be freed through us,
 * [Alias] keeps its parent alive and cannot free or resize, [Owned] storage is freed exactly once,
 * by its scope.
 */
public sealed interface Owner {
    /** We allocated the bytes; the scope of [scope] kind frees them. */
    public data class Owned(val scope: ScopeKind) : Owner
    /** The caller's array / buffer / segment / mmap; we never free it. [external] identifies the lender for debugging. */
    public data class Borrowed(val external: Any? = null) : Owner
    /** A view's storage reference: a strong reference to [parent]; mutability delegated. */
    public class Alias(public val parent: Storage) : Owner {
        override fun toString(): String = "Alias(parent=${parent.id})"
    }
}

/** Thrown on any access to a storage after its scope or the storage itself was closed (SKEEP-003 rule 2). */
public class StorageClosedException(
    public val storageId: StorageId,
    public val origin: TensorId?,
    message: String = "Storage ${storageId}${origin?.let { " (" + it.canonical + ")" } ?: ""} is closed",
) : IllegalStateException(message)

/**
 * The one and only owner of bytes (SKEEP-003 §0, §4.2). `TensorView` interprets a storage, `Tensor`
 * is the DSL handle over a view — neither owns bytes. Sealed over the four kinds; [Heap] is final
 * and common, [OffHeap] / [Mapped] / [Device] are abstract here and bound per platform
 * (`MemorySegment` / `FileChannel.map` on the JVM, `malloc` / `mmap` on Native, heap fallbacks on
 * JS/Wasm — slices #1019, #1020).
 *
 * Rules enforced here: exactly one byte owner; closing invalidates every alias; a borrowed storage
 * is released (forgotten) but never freed; every access after close throws
 * [StorageClosedException] carrying the id and origin — not a JVM crash, not silent corruption.
 */
public sealed class Storage : AutoCloseable {
    public abstract val id: StorageId
    public abstract val sizeBytes: Long
    public abstract val owner: Owner
    public abstract val domain: MemoryDomain
    /** The `TensorId` these bytes back, for diagnostics; `null` for anonymous storage. */
    public abstract val debugOrigin: TensorId?
    /** Where trace events about this storage go (allocation, close). */
    protected abstract val sink: TraceSink

    /** The lifetime class: from [Owner.Owned], else the parent's, else `AMBIENT`. */
    public val scope: ScopeKind
        get() = when (val o = owner) {
            is Owner.Owned -> o.scope
            is Owner.Alias -> o.parent.scope
            is Owner.Borrowed -> ScopeKind.AMBIENT
        }

    private var closed: Boolean = false

    /** `true` until this storage — or, for an alias, its parent — is closed. */
    public val isAlive: Boolean
        get() = !closed && ((owner as? Owner.Alias)?.parent?.isAlive ?: true)

    /** Whether writes are allowed: owned and borrowed-mutable storage yes; an alias delegates to its parent. */
    public abstract val isMutable: Boolean

    /** Throws [StorageClosedException] if this storage is no longer alive. Called by every accessor. */
    public fun checkAlive() {
        if (isAlive) return
        // In debug mode the ledger knows where this storage was allocated and closed, which is the
        // difference between "something is closed" and "this weight was freed at model.close()".
        val detail = if (MemoryDebug.isEnabled) "\n" + MemoryDebug.describeClosed(id) else ""
        throw StorageClosedException(id, debugOrigin, "Storage $id${debugOrigin?.let { " (" + it.canonical + ")" } ?: ""} is closed$detail")
    }

    /**
     * Close: an [Owner.Owned] storage releases its bytes (exactly once); an [Owner.Borrowed] storage
     * is forgotten (the lender's bytes are untouched); an [Owner.Alias] is detached (its parent is
     * unaffected). Idempotent.
     */
    final override fun close() {
        if (closed) return
        closed = true
        onClose()
        if (owner !is Owner.Alias) {
            if (sink.isEnabled) sink.emit(TraceEvent.Free(id.value, scope, sizeBytes))
            MemoryDebug.recordClose(id, if (MemoryDebug.isEnabled) platformCallSite() else null)
        }
    }

    /** Release platform resources (owned storage only); default nothing. */
    protected open fun onClose() {}

    /** A zero-copy alias over `[offsetBytes, offsetBytes + lengthBytes)` of this storage. */
    public abstract fun slice(offsetBytes: Long, lengthBytes: Long): Storage

    /**
     * Copy [length] bytes starting at [offset] into [dest] starting at [destOffset] — the one
     * primitive that reads packed bytes uniformly regardless of storage kind (#1202). A kernel or
     * decoder that only has a [ByteArray]-shaped contract (native FFM/JNI calls, [TernaryCodec])
     * uses this to get a transient snapshot out of [OffHeap]/[Mapped] storage; for [Heap] it is a
     * plain array copy, no snapshot needed.
     */
    public abstract fun copyInto(dest: ByteArray, destOffset: Int = 0, offset: Long = 0, length: Int = dest.size - destOffset)

    /**
     * Copy [length] bytes from [src] starting at [srcOffset] into this storage starting at
     * [offset] — the inverse of [copyInto], e.g. writing a repacked payload into freshly
     * allocated off-heap storage. Requires [isMutable].
     */
    public abstract fun copyFrom(src: ByteArray, srcOffset: Int = 0, offset: Long = 0, length: Int = src.size - srcOffset)

    override fun toString(): String = "${this::class.simpleName}(${id}, ${sizeBytes} B, $owner, $domain${debugOrigin?.let { ", $it" } ?: ""}${if (isAlive) "" else ", closed"})"

    /**
     * Heap storage: a Kotlin array on the managed heap — the JIT-friendliest kind, the only kind on
     * JS/Wasm, the default for `Ambient` scope. Exactly one of [floats], [ints], [bytes] is non-null;
     * [arrayOffset] (in elements of that array) and [sizeBytes] delimit the region.
     *
     * Kernels unwrap once per call (`floats` / `ints` / `bytes` + [arrayOffset]) — the Phase-2 spike
     * showed per-element access through a view is the slow path by design.
     */
    public class Heap private constructor(
        override val id: StorageId,
        public val floats: FloatArray?,
        public val ints: IntArray?,
        public val bytes: ByteArray?,
        public val arrayOffset: Int,
        override val sizeBytes: Long,
        override val owner: Owner,
        override val debugOrigin: TensorId?,
        override val sink: TraceSink,
        private val mutable: Boolean,
    ) : Storage() {
        override val domain: MemoryDomain get() = MemoryDomain.HOST_HEAP
        override val isMutable: Boolean get() = (owner as? Owner.Alias)?.parent?.isMutable ?: mutable

        /** Bytes per element of the backing array (4 for floats/ints, 1 for bytes). */
        public val elementBytes: Int get() = if (bytes != null) 1 else 4
        /** Number of array elements this storage spans. */
        public val elementCount: Int get() = (sizeBytes / elementBytes).toInt()

        override fun slice(offsetBytes: Long, lengthBytes: Long): Heap {
            checkAlive()
            require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) { "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes" }
            require(offsetBytes % elementBytes == 0L && lengthBytes % elementBytes == 0L) { "slice must align to $elementBytes-byte elements" }
            return Heap(StorageId.next(), floats, ints, bytes, arrayOffset + (offsetBytes / elementBytes).toInt(), lengthBytes, Owner.Alias(this), debugOrigin, sink, mutable)
        }

        override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
            checkAlive()
            val b = bytes ?: throw UnsupportedOperationException("copyInto needs byte-backed Heap storage")
            val start = arrayOffset + offset.toInt()
            b.copyInto(dest, destOffset, start, start + length)
        }

        override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
            checkAlive()
            require(isMutable) { "storage $id is not mutable" }
            val b = bytes ?: throw UnsupportedOperationException("copyFrom needs byte-backed Heap storage")
            val start = arrayOffset + offset.toInt()
            src.copyInto(b, start, srcOffset, srcOffset + length)
        }

        public companion object {
            private fun create(floats: FloatArray?, ints: IntArray?, bytes: ByteArray?, offset: Int, count: Int, owner: Owner, origin: TensorId?, sink: TraceSink, mutable: Boolean): Heap {
                val eb = if (bytes != null) 1 else 4
                val s = Heap(StorageId.next(), floats, ints, bytes, offset, count.toLong() * eb, owner, origin, sink, mutable)
                if (owner is Owner.Owned) {
                    val site = if (MemoryDebug.isEnabled) platformCallSite() else null
                    if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, owner.scope, s.sizeBytes, origin, site))
                    MemoryDebug.recordAllocation(s.id, owner.scope, s.sizeBytes, origin, site)
                }
                return s
            }

            /** Allocate [count] zeroed floats on the heap, owned by a scope of kind [scope]. */
            public fun floats(count: Int, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(FloatArray(count), null, null, 0, count, Owner.Owned(scope), origin, sink, true)
            public fun ints(count: Int, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(null, IntArray(count), null, 0, count, Owner.Owned(scope), origin, sink, true)
            public fun bytes(count: Int, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(null, null, ByteArray(count), 0, count, Owner.Owned(scope), origin, sink, true)

            /** Wrap the caller's array without copying — never freed by us (the #782 `copyOf` replacement). */
            public fun wrap(array: FloatArray, offset: Int = 0, count: Int = array.size - offset, mutable: Boolean = true, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(array, null, null, offset, count, Owner.Borrowed(array), origin, sink, mutable)
            public fun wrap(array: IntArray, offset: Int = 0, count: Int = array.size - offset, mutable: Boolean = true, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(null, array, null, offset, count, Owner.Borrowed(array), origin, sink, mutable)
            public fun wrap(array: ByteArray, offset: Int = 0, count: Int = array.size - offset, mutable: Boolean = true, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Heap =
                create(null, null, array, offset, count, Owner.Borrowed(array), origin, sink, mutable)
        }
    }

    /** Off-heap storage (`MemorySegment` / direct buffer / `malloc`): bound per platform in #1019/#1020. */
    public abstract class OffHeap : Storage() { override val domain: MemoryDomain get() = MemoryDomain.HOST_OFFHEAP }

    /** A mapped file region (`FileChannel.map` / `mmap`): bound per platform in #1019/#1020. */
    public abstract class Mapped : Storage() { override val domain: MemoryDomain get() = MemoryDomain.MMAP_FILE }

    /** An accelerator buffer — placeholder until a device backend is scheduled (PRD non-goal). */
    public abstract class Device : Storage() { override val domain: MemoryDomain get() = MemoryDomain.DEVICE_LOCAL }
}
