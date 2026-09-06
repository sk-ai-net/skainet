package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.MemoryDomain

/**
 * The lifetime class an allocation belongs to (SKEEP-003 §4.5). `Scope` itself — the object that
 * allocates and frees — arrives with milestone M1; this enum is its `kind`, declared now so that
 * memory plans and allocation specs can name the lifetime without depending on the allocator.
 */
public enum class ScopeKind {
    /** Lives until the model is closed: weights, KV-cache backing, embedding tables. */
    MODEL,
    /** Recycled every forward pass: activations, attention scratch, adapter outputs. */
    FORWARD,
    /** Garbage-collected, no explicit lifetime — the default for notebooks, tests and ad-hoc tensors. */
    AMBIENT,
}

/**
 * What an allocation needs: the [Format] of the elements, how many, and where / for how long the
 * bytes should live. Pure description — owns nothing. The single input of the memory plan
 * (milestone M0) and, from M1, of `Storage.allocate(spec, scope)`.
 *
 * Replaces the never-consumed `StorageSpec` (decision recorded in SKEEP-003: "StorageSpec becomes
 * the allocation spec").
 *
 * @property format dtype + encoding of the elements
 * @property elementCount logical number of elements
 * @property domain where the bytes should be (heap, off-heap/pinned, mapped file, device …)
 * @property scope the lifetime class the allocation belongs to
 * @property mutable whether the bytes may be written after allocation
 * @property alignment required byte alignment of the start of the buffer (SIMD kernels want 16–64)
 */
public data class AllocationSpec(
    val format: Format,
    val elementCount: Long,
    val domain: MemoryDomain = MemoryDomain.HOST_HEAP,
    val scope: ScopeKind = ScopeKind.AMBIENT,
    val mutable: Boolean = true,
    val alignment: Int = DEFAULT_ALIGNMENT,
) {
    init {
        require(elementCount >= 0) { "elementCount must be >= 0, was $elementCount" }
        require(alignment > 0 && (alignment and (alignment - 1)) == 0) { "alignment must be a power of two, was $alignment" }
    }

    /** Physical bytes this allocation needs, or `null` when the encoding cannot tell (opaque payloads). */
    val bytesOrNull: Long? get() = format.physicalBytes(elementCount)

    /**
     * Physical bytes this allocation needs.
     * @throws IllegalStateException for an encoding that cannot compute its size (see [bytesOrNull])
     */
    val bytes: Long
        get() = bytesOrNull ?: throw IllegalStateException("Encoding ${format.encoding.name} cannot compute a byte size for $elementCount elements")

    public companion object {
        /** 64 bytes: satisfies AVX-512 / NEON / cache-line alignment for every current kernel. */
        public const val DEFAULT_ALIGNMENT: Int = 64

        /** Spec for a tensor of [shape] in [format]. */
        public fun of(
            format: Format,
            shape: Shape,
            domain: MemoryDomain = MemoryDomain.HOST_HEAP,
            scope: ScopeKind = ScopeKind.AMBIENT,
            mutable: Boolean = true,
            alignment: Int = DEFAULT_ALIGNMENT,
        ): AllocationSpec = AllocationSpec(format, shape.volume.toLong(), domain, scope, mutable, alignment)
    }
}
