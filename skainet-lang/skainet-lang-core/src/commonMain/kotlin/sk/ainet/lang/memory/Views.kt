package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.Tensor

/**
 * The one view mechanism (SKEEP-003 §4.6, #1034): everything that used to be a *separate* way of
 * looking at someone else's bytes — the index-remapping `SlicedTensorView`, the byte-range
 * `BufferHandle.Aliased`, the packed-transpose rewrap — is a [TensorView] with a different
 * [Layout] over the same [Storage].
 *
 * These entry points are how DSL code reaches it: [view] turns a [Tensor] into the view its data
 * already exposes, and [slice] replays the old `Slice` DSL on top of `narrow`/`step`/`squeeze`, so
 * a caller migrating off `Tensor.sliceView` keeps the same vocabulary and gets the same view type
 * every other operation returns.
 */

/** This tensor as a [TensorView] over the *same* bytes, or `null` if its data cannot expose one. */
public fun Tensor<*, *>.viewOrNull(): TensorView? = data.view

/**
 * This tensor as a [TensorView] over the *same* bytes — zero-copy.
 *
 * @throws UnsupportedOperationException if the tensor's data has no view (a backend type that owns
 *   its bytes elsewhere, e.g. device-resident data); use `copyToFloatArray()` for those.
 */
public fun Tensor<*, *>.view(): TensorView = viewOrNull() ?: throw UnsupportedOperationException(
    "${data::class.simpleName} does not expose a TensorView; it holds its bytes somewhere this milestone cannot address"
)

/**
 * The old `Slice` DSL as layout arithmetic: `Range`/`All` narrow, `Step` narrows then strides,
 * `At` narrows to one and drops the axis. One slice per axis, exactly as `Tensor.slice` required.
 *
 * The result is an ordinary [TensorView] — the same type [narrow], [transpose], [unsqueeze] and
 * [squeeze] return, which is the whole point of #1034.
 */
public fun TensorView.slice(slices: List<Slice<*, *>>): TensorView {
    require(slices.size == shape.rank) {
        "expected one slice per axis (${shape.rank}), got ${slices.size}"
    }
    var v = this
    val dropped = ArrayList<Int>()
    slices.forEachIndexed { axis, raw ->
        val extent = shape[axis]
        require(raw.isValid(extent)) { "invalid slice for axis $axis (extent $extent): $raw" }
        when (val s = raw.normalize(extent)) {
            is Slice.All -> Unit
            is Slice.Range -> v = v.narrow(axis, s.start, s.end - s.start)
            is Slice.At -> { v = v.narrow(axis, s.index, 1); dropped += axis }
            is Slice.Step -> v = v.narrow(axis, s.start, s.end - s.start).step(axis, s.step)
        }
    }
    // Highest axis first, so the earlier indices stay valid while axes disappear.
    for (axis in dropped.sortedDescending()) v = v.squeeze(axis)
    return v
}

/** [slice] with the slices spelled out positionally. */
public fun TensorView.slice(vararg slices: Slice<*, *>): TensorView = slice(slices.toList())
