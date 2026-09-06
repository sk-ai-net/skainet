package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * JS / Wasm: linear memory only — no off-heap, no mmap (SKEEP-003 §4.8.4). Off-heap requests resolve
 * to the heap (the planner records the fallback); files arrive through fetch/range requests into
 * heap slabs, so [mapFile] is unsupported here.
 */
public actual object PlatformStorage {
    public actual fun supports(domain: MemoryDomain): Boolean = domain == MemoryDomain.HOST_HEAP
    public actual val supportsMappedFiles: Boolean get() = false

    public actual fun allocate(bytes: Long, domain: MemoryDomain, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage {
        require(bytes in 0..Int.MAX_VALUE.toLong()) { "heap storage is limited to 2 GB per array, requested $bytes bytes" }
        return when (domain) {
            MemoryDomain.MMAP_FILE, MemoryDomain.DEVICE_LOCAL -> throw IllegalArgumentException("$domain is not allocatable on this target")
            else -> Storage.Heap.bytes(bytes.toInt(), scope, origin, sink) // HOST_OFFHEAP / PINNED / UNIFIED fall back to the heap
        }
    }

    public actual fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
        throw UnsupportedOperationException("memory-mapped files are not available on this target (use the suspend RandomAccessSource into heap slabs)")

    public val info: PlatformStorageInfo = PlatformStorageInfo("heap (fallback)", "none")
}
