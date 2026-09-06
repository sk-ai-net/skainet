package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.tensor.storage.TensorStorage
import sk.ainet.lang.types.DType

/**
 * The pair `(dtype, encoding)` — **what** a value means and **how** its bytes are laid out.
 *
 * A Q4_K weight is `Format(FP32, TensorEncoding.Q4_K)`: logically FP32, stored as Q4_K blocks. A
 * plain float tensor is `Format(FP32, TensorEncoding.Dense(4))`. Kernel dispatch keys on formats
 * (SKEEP-003 §0, §5); rule 3 — the logical dtype is never erased by a packed encoding.
 *
 * `Format` is pure metadata: it owns no bytes and carries no shape.
 */
public data class Format(val dtype: DType, val encoding: TensorEncoding) {

    /** True when the bytes are the dtype's own dense representation (no block packing). */
    val isDense: Boolean get() = encoding is TensorEncoding.Dense

    /** Physical bytes for [elementCount] elements under this format, or `null` if the encoding cannot tell. */
    public fun physicalBytes(elementCount: Long): Long? = encoding.physicalBytes(elementCount)

    /** `Float32/Q4_K`, `Float32/Dense(4B)` — the form the `toString()` renderer prints. */
    override fun toString(): String = "${dtype.name}/${encoding.name}"

    public companion object {
        /** The dense format of [dtype] at its own width (`Dense(dtype.sizeInBytes)`). */
        public fun dense(dtype: DType): Format = Format(dtype, TensorEncoding.Dense(dtype.sizeInBytes))
    }
}

/**
 * The [Format] of this tensor: its [Tensor.dtype] witness mapped back to a [DType] plus the
 * encoding its data reports ([sk.ainet.lang.tensor.data.TensorData.encoding], dense at the
 * dtype's width when the data reports none).
 *
 * @throws IllegalStateException if [Tensor.dtype] is not a concrete dtype class (e.g. `DType::class`)
 */
public val Tensor<*, *>.format: Format
    get() = formatOrNull
        ?: throw IllegalStateException("Tensor.dtype ${this.dtype} is not a concrete DType witness; cannot derive a Format")

/** The [Format] of this tensor, or `null` if its dtype witness is not a concrete dtype class. */
public val Tensor<*, *>.formatOrNull: Format?
    get() {
        val dt = DType.fromWitnessOrNull(this.dtype) ?: return null
        return Format(dt, this.data.encoding ?: TensorEncoding.Dense(dt.sizeInBytes))
    }

/** The [Format] of this storage descriptor: `(dtype, encoding)`. */
public val TensorStorage.format: Format
    get() = Format(dtype, encoding)
