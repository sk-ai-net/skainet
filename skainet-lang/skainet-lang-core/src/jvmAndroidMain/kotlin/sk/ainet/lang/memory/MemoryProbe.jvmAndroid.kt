package sk.ainet.lang.memory

import java.io.File

/**
 * JVM and Android: `/proc/self` — the kernel's own numbers, not the JVM's.
 *
 * `Runtime.totalMemory()` describes the managed heap, which is exactly the thing that is *not*
 * interesting once weights live in mapped pages. `statm` field 2 is the resident page count and
 * `stat` fields 10 and 12 are the minor and major fault counters (see `proc(5)`).
 *
 * Returns `null` off Linux (macOS, Windows), where these files do not exist.
 */
public actual object MemoryProbe {

    private val pageSize: Long = 4096L

    public actual fun rssBytes(): Long? {
        val fields = read("/proc/self/statm")?.split(" ") ?: return null
        val pages = fields.getOrNull(1)?.trim()?.toLongOrNull() ?: return null
        return pages * pageSize
    }

    public actual fun majorFaults(): Long? = statField(12)

    public actual fun minorFaults(): Long? = statField(10)

    /** `/proc/self/stat` is 1-indexed in `proc(5)`; the comm field may contain spaces, so split after it. */
    private fun statField(index: Int): Long? {
        val stat = read("/proc/self/stat") ?: return null
        val afterComm = stat.substringAfterLast(") ")
        val fields = afterComm.split(" ")
        // field 3 (state) is the first token after comm, so `index` maps to `index - 3`
        return fields.getOrNull(index - 3)?.trim()?.toLongOrNull()
    }

    private fun read(path: String): String? = try {
        val file = File(path)
        if (file.canRead()) file.readText().trim() else null
    } catch (e: Exception) {
        null
    }
}
