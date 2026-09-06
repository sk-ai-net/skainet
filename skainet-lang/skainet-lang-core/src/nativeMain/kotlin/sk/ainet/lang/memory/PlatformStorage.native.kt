package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain

/** Kotlin/Native: off-heap = `malloc` ([NativeMallocStorage]), mapped = `mmap` ([NativeMappedStorage]). */
public actual object PlatformStorage {
    public actual fun supports(domain: MemoryDomain): Boolean = domain == MemoryDomain.HOST_HEAP || domain == MemoryDomain.HOST_OFFHEAP || domain == MemoryDomain.MMAP_FILE
    public actual val supportsMappedFiles: Boolean get() = true

    public actual fun allocate(bytes: Long, domain: MemoryDomain, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage = when (domain) {
        MemoryDomain.HOST_OFFHEAP, MemoryDomain.HOST_PINNED, MemoryDomain.UNIFIED -> NativeMallocStorage.allocate(bytes, scope, origin, sink)
        MemoryDomain.HOST_HEAP -> {
            require(bytes in 0..Int.MAX_VALUE.toLong()) { "heap storage is limited to 2 GB per array, requested $bytes bytes" }
            Storage.Heap.bytes(bytes.toInt(), scope, origin, sink)
        }
        MemoryDomain.MMAP_FILE, MemoryDomain.DEVICE_LOCAL -> throw IllegalArgumentException("$domain is not allocatable; use mapFile / a device backend")
    }

    public actual fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
        NativeMappedStorage.map(path, fileOffset, length, scope, origin, sink)

    public val info: PlatformStorageInfo = PlatformStorageInfo("malloc", "mmap")
}
