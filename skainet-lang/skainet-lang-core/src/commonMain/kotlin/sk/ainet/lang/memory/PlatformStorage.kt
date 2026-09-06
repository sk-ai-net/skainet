package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * The platform's storage kinds behind one common door (SKEEP-003 §4.8): *where bytes live is a
 * policy decision made by the planner; how long they live is a scope decision; neither is made by
 * a layer, a loader or a kernel.* Callers ask for a [MemoryDomain]; the platform binds it to what
 * it has — `MemorySegment` / `FileChannel.map` on the JVM, direct `ByteBuffer` / `MappedByteBuffer`
 * on Android, `malloc` / `mmap` on Kotlin/Native, and the heap on JS/Wasm (no off-heap, no mmap:
 * a request for those resolves to [Storage.Heap] and [PlatformStorage.supports] says so, so the
 * planner can note the fallback).
 */
public expect object PlatformStorage {
    /** Whether this target can honour [domain] natively (false = [allocate] falls back to the heap). */
    public fun supports(domain: MemoryDomain): Boolean

    /** Whether this target can map files ([mapFile] throws [UnsupportedOperationException] otherwise). */
    public val supportsMappedFiles: Boolean

    /**
     * Allocate [bytes] zeroed bytes in [domain] (or the closest the platform has), owned by a scope
     * of kind [scope]. `HOST_HEAP` → [Storage.Heap.bytes]; `HOST_OFFHEAP` → the platform off-heap kind
     * or the heap fallback; other domains are not allocatable here.
     */
    public fun allocate(bytes: Long, domain: MemoryDomain = MemoryDomain.HOST_HEAP, scope: ScopeKind = ScopeKind.AMBIENT, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Storage

    /** Map `[fileOffset, fileOffset + length)` of the file at [path] read-only into `MODEL`-scoped storage. */
    public fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind = ScopeKind.MODEL, origin: TensorId? = null, sink: TraceSink = NoopTraceSink): Storage
}

/** The kinds a target binds, for diagnostics (`describe()`, the planner's notes). */
public data class PlatformStorageInfo(val offHeap: String, val mapped: String) {
    override fun toString(): String = "OffHeap=$offHeap · Mapped=$mapped"
}
