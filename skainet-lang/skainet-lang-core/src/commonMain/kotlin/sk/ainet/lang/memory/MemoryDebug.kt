package sk.ainet.lang.memory

import sk.ainet.lang.tensor.TensorId

/**
 * Debug mode for the memory model (SKEEP-003 §4.7, §8 item 4, PRD M1-F9): the switch that turns
 * `SKAINET_MEMORY_DEBUG=1` (or `-Dskainet.memory.debug=true`) into allocation-site tagging,
 * use-after-close reporting with the closing site, adapter logging and per-scope high-water marks.
 *
 * *The tool that would have found #782 in one run.* Off by default and costing one boolean read
 * per allocation; on, it keeps a bounded ledger of live storages plus hooks a debugger or a test
 * can break on.
 */
public object MemoryDebug {

    /** Environment variable and system property that enable debug mode. */
    public const val ENV: String = "SKAINET_MEMORY_DEBUG"
    public const val PROPERTY: String = "skainet.memory.debug"

    /** Explicit override; `null` means "read the platform setting". Tests set this. */
    public var overrideEnabled: Boolean? = null

    /** Whether debug mode is on. */
    public val isEnabled: Boolean get() = overrideEnabled ?: platformMemoryDebugEnabled()

    /** What a storage's ledger entry remembers. */
    public data class Entry(
        val storageId: StorageId,
        val scope: ScopeKind,
        val bytes: Long,
        val origin: TensorId?,
        val allocationSite: String?,
        var closingSite: String? = null,
        var closed: Boolean = false,
    )

    private val ledger = LinkedHashMap<Long, Entry>()
    private val peakByScope = HashMap<ScopeKind, Long>()
    private val liveByScope = HashMap<ScopeKind, Long>()

    // --- hooks (SKEEP-003 §4.7 "breakpoint-like hooks") ---

    /** Called on every allocation while debug mode is on; throw from it to break in a debugger. */
    public var onAllocate: ((Entry) -> Unit)? = null

    /** Called when a storage is closed. */
    public var onClose: ((Entry) -> Unit)? = null

    /** Called when the dispatcher inserts an adapter; `bytes` is what it allocated. */
    public var onAdapter: ((kind: String, bytes: Long, target: TensorId?) -> Unit)? = null

    /** Called when a `Forward` scope is reset with storages still referenced from outside — a leak. */
    public var onLeak: ((List<Entry>) -> Unit)? = null

    // --- recording (called by Storage / Scope / the dispatcher; no-ops when disabled) ---

    internal fun recordAllocation(id: StorageId, scope: ScopeKind, bytes: Long, origin: TensorId?, site: String?) {
        if (!isEnabled) return
        val e = Entry(id, scope, bytes, origin, site)
        ledger[id.value] = e
        val live = (liveByScope[scope] ?: 0L) + bytes
        liveByScope[scope] = live
        if (live > (peakByScope[scope] ?: 0L)) peakByScope[scope] = live
        onAllocate?.invoke(e)
    }

    internal fun recordClose(id: StorageId, site: String?) {
        if (!isEnabled) return
        val e = ledger[id.value] ?: return
        if (e.closed) return
        e.closed = true; e.closingSite = site
        liveByScope[e.scope] = ((liveByScope[e.scope] ?: 0L) - e.bytes).coerceAtLeast(0L)
        onClose?.invoke(e)
    }

    /** Report an adapter insertion (kind, bytes, target) — the "silent dequant budget" of #782. */
    public fun recordAdapter(kind: String, bytes: Long, target: TensorId? = null) {
        if (!isEnabled) return
        onAdapter?.invoke(kind, bytes, target)
    }

    /** Report storages that outlived a `Forward` reset; the hook fires and the entries are returned. */
    public fun reportLeaks(stillReferenced: List<StorageId>): List<Entry> {
        if (!isEnabled || stillReferenced.isEmpty()) return emptyList()
        val entries = stillReferenced.mapNotNull { ledger[it.value] }.filterNot { it.closed }
        if (entries.isNotEmpty()) onLeak?.invoke(entries)
        return entries
    }

    // --- inspection ---

    /** The ledger entry of a storage, if debug mode saw it. */
    public fun entry(id: StorageId): Entry? = ledger[id.value]

    /** Storages allocated and not yet closed. */
    public fun liveEntries(): List<Entry> = ledger.values.filterNot { it.closed }

    /** Per-scope high-water mark since the last [reset]. */
    public fun peakBytes(scope: ScopeKind): Long = peakByScope[scope] ?: 0L

    /** Live bytes per scope right now. */
    public fun liveBytes(scope: ScopeKind): Long = liveByScope[scope] ?: 0L

    /**
     * A use-after-close message that names the closing site — the difference between "something is
     * closed" and "this weight was freed at model.close(), you are reading it from the KV path".
     */
    public fun describeClosed(id: StorageId): String {
        val e = ledger[id.value] ?: return "storage $id (no debug record; set $ENV=1 before allocating)"
        return buildString {
            append("storage "); append(id)
            e.origin?.let { append(" ("); append(it.canonical); append(')') }
            append(" of "); append(e.bytes); append(" B in "); append(e.scope.name.lowercase()); append(" scope")
            e.allocationSite?.let { append("\n  allocated at "); append(it) }
            e.closingSite?.let { append("\n  closed at "); append(it) }
        }
    }

    /** Everything the ledger knows, for a debugger watch window or a failing test. */
    public fun report(): String = buildString {
        append("SKaiNET memory debug — "); append(ledger.size); append(" storages seen, ")
        append(liveEntries().size); append(" live\n")
        for (scope in ScopeKind.entries) {
            val peak = peakBytes(scope); val live = liveBytes(scope)
            if (peak == 0L && live == 0L) continue
            append("  "); append(scope.name.lowercase().padEnd(8))
            append("live "); append(live); append(" B, peak "); append(peak); append(" B\n")
        }
        for (e in liveEntries().take(MAX_REPORTED)) {
            append("  live "); append(e.storageId); append(' ')
            append(e.origin?.canonical ?: "—"); append(' '); append(e.bytes); append(" B")
            e.allocationSite?.let { append(" @ "); append(it) }
            append('\n')
        }
        if (liveEntries().size > MAX_REPORTED) { append("  … "); append(liveEntries().size - MAX_REPORTED); append(" more\n") }
    }

    /** Forget everything (between tests, or between models). */
    public fun reset() {
        ledger.clear(); peakByScope.clear(); liveByScope.clear()
    }

    private const val MAX_REPORTED: Int = 20
}

/** Platform reading of [MemoryDebug.ENV] / [MemoryDebug.PROPERTY]. */
internal expect fun platformMemoryDebugEnabled(): Boolean

/** The current call site, when the platform can cheaply produce one (JVM: a stack frame). */
internal expect fun platformCallSite(): String?
