package sk.ainet.lang.memory

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.refTo
import kotlinx.cinterop.toKString
import platform.posix.fclose
import platform.posix.fgets
import platform.posix.fopen

/**
 * Kotlin/Native on Linux: the same `/proc/self` files the JVM actual reads. On a platform without
 * them (macOS, iOS) `fopen` fails and every value is `null`, which is the honest answer.
 */
@OptIn(ExperimentalForeignApi::class)
public actual object MemoryProbe {

    private const val PAGE_SIZE: Long = 4096L

    public actual fun rssBytes(): Long? {
        val fields = read("/proc/self/statm")?.split(" ") ?: return null
        val pages = fields.getOrNull(1)?.trim()?.toLongOrNull() ?: return null
        return pages * PAGE_SIZE
    }

    public actual fun majorFaults(): Long? = statField(12)

    public actual fun minorFaults(): Long? = statField(10)

    private fun statField(index: Int): Long? {
        val stat = read("/proc/self/stat") ?: return null
        val afterComm = stat.substringAfterLast(") ")
        return afterComm.split(" ").getOrNull(index - 3)?.trim()?.toLongOrNull()
    }

    private fun read(path: String): String? {
        val file = fopen(path, "r") ?: return null
        try {
            val buffer = ByteArray(4096)
            val line = fgets(buffer.refTo(0), buffer.size, file) ?: return null
            return line.toKString().trim()
        } finally {
            fclose(file)
        }
    }
}
