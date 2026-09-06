package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * A lifetime that owned storage belongs to (SKEEP-003 §0 *Scope*, §4.5). Three kinds:
 * [Scope.Ambient] (GC-managed — the default, today's behaviour), [ForwardScope] (activations;
 * recycled every forward pass by [ForwardScope.reset]), [ModelScope] (weights, KV backing; closed
 * on `model.close()`). Closing a scope invalidates every storage it owns (rule 2); scopes are
 * opt-in per `ExecutionContext` so `val c = a matMul b` in a notebook keeps working unchanged.
 *
 * Both historical arena failures were violations of exactly this split: activations in a
 * model-lifetime arena (tens of GB pinned) and per-call arenas (leak per matmul). A `Forward`
 * scope matches the forward-pass lifetime; a `Model` scope the model's.
 */
public sealed interface Scope : AutoCloseable {
    public val kind: ScopeKind
    /** Bytes currently owned and alive in this scope. */
    public val liveBytes: Long
    /** Where this scope's allocation / reset events go. */
    public val sink: TraceSink

    /** Allocate [bytes] zeroed bytes in [domain] (platform-bound), owned by this scope. */
    public fun allocate(bytes: Long, domain: MemoryDomain = MemoryDomain.HOST_HEAP, origin: TensorId? = null): Storage

    /** Allocate [count] zeroed floats on the heap, owned by this scope (the kernel-friendliest form). */
    public fun allocateFloats(count: Int, origin: TensorId? = null): Storage.Heap

    /** The GC-managed default: nothing is tracked, nothing is freed explicitly; [close] and [liveBytes] are no-ops. */
    public object Ambient : Scope {
        override val kind: ScopeKind get() = ScopeKind.AMBIENT
        override val liveBytes: Long get() = 0L
        override val sink: TraceSink get() = NoopTraceSink
        override fun allocate(bytes: Long, domain: MemoryDomain, origin: TensorId?): Storage = PlatformStorage.allocate(bytes, domain, ScopeKind.AMBIENT, origin, NoopTraceSink)
        override fun allocateFloats(count: Int, origin: TensorId?): Storage.Heap = Storage.Heap.floats(count, ScopeKind.AMBIENT, origin, NoopTraceSink)
        override fun close() {}
    }
}

/**
 * `Scope.Model`: weights, KV-cache backing, embedding tables — everything that lives until the
 * model is closed. Tracks every storage it allocates or maps and closes them all in [close]
 * (deterministic: the JVM unmaps the file, native `free`s/`munmap`s). Idiomatic use:
 * `ModelScope().use { model -> … }`.
 */
public class ModelScope(override val sink: TraceSink = NoopTraceSink, public val name: String = "model") : Scope {
    override val kind: ScopeKind get() = ScopeKind.MODEL
    private val owned = ArrayList<Storage>()
    private var closed = false
    public val isClosed: Boolean get() = closed

    override val liveBytes: Long get() = owned.sumOf { if (it.isAlive) it.sizeBytes else 0L }
    /** Number of storages this scope owns (alive or not). */
    public val storageCount: Int get() = owned.size

    private fun track(s: Storage): Storage { check(!closed) { "ModelScope '$name' is closed" }; owned += s; return s }

    override fun allocate(bytes: Long, domain: MemoryDomain, origin: TensorId?): Storage = track(PlatformStorage.allocate(bytes, domain, ScopeKind.MODEL, origin, sink))
    override fun allocateFloats(count: Int, origin: TensorId?): Storage.Heap = track(Storage.Heap.floats(count, ScopeKind.MODEL, origin, sink)) as Storage.Heap

    /** Map a file region (packed weights) into this scope; unmapped when the scope closes. */
    public fun mapFile(path: String, fileOffset: Long, length: Long, origin: TensorId? = null): Storage = track(PlatformStorage.mapFile(path, fileOffset, length, ScopeKind.MODEL, origin, sink))

    /**
     * Adopt a storage allocated elsewhere (a loader's mapped weight, a borrowed array) so the scope
     * closes it — and **count it**: this is the point where the model takes responsibility for those
     * bytes. Mapped and borrowed weights emit no allocation event of their own (borrowing is not an
     * allocation), yet decision #11 counts weights as resident because decode touches every weight
     * every token, so without this the memory plan would be checked against a run that appears to
     * hold nothing (#1074).
     */
    public fun adopt(storage: Storage): Storage {
        val s = track(storage)
        if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, ScopeKind.MODEL, s.sizeBytes, s.debugOrigin, site = "adopted"))
        return s
    }

    /** Close every owned storage (weights unmapped, off-heap freed) — exactly once. */
    override fun close() {
        if (closed) return
        closed = true
        for (s in owned.asReversed()) {
            val wasBorrowed = s.owner is Owner.Borrowed
            val bytes = s.sizeBytes
            val id = s.id
            s.close()
            // A borrowed storage does not emit Free on its own (we never freed anything), but the
            // scope did stop counting it — keep the ledger balanced for plan-vs-actual.
            if (wasBorrowed && sink.isEnabled) sink.emit(TraceEvent.Free(id.value, ScopeKind.MODEL, bytes))
        }
        owned.clear()
    }
}

/**
 * `Scope.Forward`: activations, attention scratch, adapter outputs — recycled every forward pass.
 * One pre-sized slab ([slabFloats] floats on the heap, per the Phase-2 spike: heap activations on
 * the JVM by default) is bump-allocated; [reset] at the end of the step rewinds the offset and
 * invalidates the views handed out, so steady-state decode allocates zero slab bytes. If a step
 * needs more than the slab, an overflow storage is allocated (tracked, closed at [reset], counted
 * in [overflowBytes] so the planner can resize the slab). Outputs that must outlive the step are
 * copied out with [retain].
 */
public class ForwardScope(
    public val slabFloats: Int,
    override val sink: TraceSink = NoopTraceSink,
    public val name: String = "forward",
) : Scope {
    init { require(slabFloats >= 0) { "slabFloats must be >= 0" } }

    override val kind: ScopeKind get() = ScopeKind.FORWARD

    private val slab: Storage.Heap = Storage.Heap.floats(slabFloats, ScopeKind.FORWARD, TensorId(listOf(name), "slab"), sink)
    private var offset: Int = 0
    private val handedOut = ArrayList<Storage>()
    private val overflow = ArrayList<Storage>()
    private var closed = false

    /** Floats allocated from the slab in the current step. */
    public val usedFloats: Int get() = offset
    /** High-water mark of slab use across steps (floats). */
    public var peakFloats: Int = 0
        private set
    /** Bytes allocated outside the slab in the current step (the planner should grow the slab by this). */
    public val overflowBytes: Long get() = overflow.sumOf { it.sizeBytes }
    /** Steps completed ([reset] calls). */
    public var steps: Long = 0L
        private set
    public val isClosed: Boolean get() = closed

    override val liveBytes: Long get() = offset.toLong() * 4 + overflowBytes

    private fun checkOpen() { check(!closed) { "ForwardScope '$name' is closed" } }

    /** Bump-allocate [count] floats from the slab (or an overflow storage when the slab is exhausted). */
    override fun allocateFloats(count: Int, origin: TensorId?): Storage.Heap {
        checkOpen(); require(count >= 0)
        if (offset + count <= slabFloats) {
            val view = slab.slice(offset.toLong() * 4, count.toLong() * 4)
            offset += count
            if (offset > peakFloats) peakFloats = offset
            handedOut += view
            return view
        }
        val extra = Storage.Heap.floats(count, ScopeKind.FORWARD, origin, sink)
        overflow += extra
        return extra
    }

    override fun allocate(bytes: Long, domain: MemoryDomain, origin: TensorId?): Storage {
        checkOpen()
        if (domain == MemoryDomain.HOST_HEAP && bytes % 4 == 0L && bytes / 4 <= Int.MAX_VALUE) return allocateFloats((bytes / 4).toInt(), origin)
        val s = PlatformStorage.allocate(bytes, domain, ScopeKind.FORWARD, origin, sink)
        overflow += s
        return s
    }

    /**
     * Copy a step-scoped storage out into a [to] scope (default [Scope.Ambient]) so it survives
     * [reset] — the one sanctioned escape (the rest is a use-after-reset, caught as
     * `StorageClosedException`).
     */
    public fun retain(storage: Storage.Heap, to: Scope = Scope.Ambient, origin: TensorId? = storage.debugOrigin): Storage.Heap {
        storage.checkAlive()
        val floats = storage.floats ?: throw IllegalArgumentException("retain() supports float storage in this milestone")
        val out = to.allocateFloats(storage.elementCount, origin)
        floats.copyInto(out.floats!!, out.arrayOffset, storage.arrayOffset, storage.arrayOffset + storage.elementCount)
        return out
    }

    /** End of step: invalidate every view handed out, free overflow, rewind the slab. Emits [TraceEvent.ScopeReset]. */
    public fun reset() {
        checkOpen()
        val before = liveBytes
        for (s in handedOut) s.close()
        handedOut.clear()
        for (s in overflow) s.close()
        overflow.clear()
        offset = 0
        steps++
        if (sink.isEnabled) sink.emit(TraceEvent.ScopeReset(ScopeKind.FORWARD, before, 0L))
    }

    /** Close the slab itself (end of the model's life). */
    override fun close() {
        if (closed) return
        reset()
        closed = true
        slab.close()
    }
}
