package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import java.nio.file.Path

/** JVM: off-heap = FFM `MemorySegment` ([SegmentStorage]), mapped = `FileChannel.map` ([MappedFileStorage]). */
public actual object PlatformStorage {
    public actual fun supports(domain: MemoryDomain): Boolean = domain == MemoryDomain.HOST_HEAP || domain == MemoryDomain.HOST_OFFHEAP || domain == MemoryDomain.MMAP_FILE
    public actual val supportsMappedFiles: Boolean get() = true

    public actual fun allocate(bytes: Long, domain: MemoryDomain, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage = when (domain) {
        MemoryDomain.HOST_OFFHEAP, MemoryDomain.HOST_PINNED, MemoryDomain.UNIFIED -> SegmentStorage.allocate(bytes, scope, origin = origin, sink = sink)
        MemoryDomain.HOST_HEAP -> Storage.Heap.bytes(checkedInt(bytes), scope, origin, sink)
        MemoryDomain.MMAP_FILE, MemoryDomain.DEVICE_LOCAL -> throw IllegalArgumentException("$domain is not allocatable; use mapFile / a device backend")
    }

    public actual fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
        MappedFileStorage.map(Path.of(path), fileOffset, length, scope, origin, sink)

    public val info: PlatformStorageInfo = PlatformStorageInfo("MemorySegment (FFM)", "FileChannel.map → MemorySegment")
}

internal fun checkedInt(bytes: Long): Int {
    require(bytes in 0..Int.MAX_VALUE.toLong()) { "heap storage is limited to 2 GB per array, requested $bytes bytes" }
    return bytes.toInt()
}
