package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import java.nio.file.Path

/** Android: off-heap = direct `ByteBuffer` (outside the ART heap limit, #922), mapped = `MappedByteBuffer` (SKEEP-002 / #921). */
public actual object PlatformStorage {
    public actual fun supports(domain: MemoryDomain): Boolean = domain == MemoryDomain.HOST_HEAP || domain == MemoryDomain.HOST_OFFHEAP || domain == MemoryDomain.MMAP_FILE
    public actual val supportsMappedFiles: Boolean get() = true

    public actual fun allocate(bytes: Long, domain: MemoryDomain, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage = when (domain) {
        MemoryDomain.HOST_OFFHEAP, MemoryDomain.HOST_PINNED, MemoryDomain.UNIFIED -> DirectBufferStorage.allocate(checkedInt(bytes), scope, origin, sink)
        MemoryDomain.HOST_HEAP -> Storage.Heap.bytes(checkedInt(bytes), scope, origin, sink)
        MemoryDomain.MMAP_FILE, MemoryDomain.DEVICE_LOCAL -> throw IllegalArgumentException("$domain is not allocatable; use mapFile / a device backend")
    }

    public actual fun mapFile(path: String, fileOffset: Long, length: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
        MappedBufferStorage.map(Path.of(path), fileOffset, length, scope, origin, sink)

    public val info: PlatformStorageInfo = PlatformStorageInfo("direct ByteBuffer", "FileChannel.map → MappedByteBuffer")
}

internal fun checkedInt(bytes: Long): Int {
    require(bytes in 0..Int.MAX_VALUE.toLong()) { "buffer storage is limited to 2 GB per buffer, requested $bytes bytes" }
    return bytes.toInt()
}
