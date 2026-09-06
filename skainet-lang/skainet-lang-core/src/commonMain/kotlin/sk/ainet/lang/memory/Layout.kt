package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape

/**
 * Which order a block-packed buffer's blocks are in — the contract #973 reports as unwritten and
 * contradictory across the engine and its converters.
 *
 * Two orders exist for a 2-D `[out, in]` weight whose blocks tile the input dimension:
 *
 * - [ROW_MAJOR] — flat block index `o * blocksPerRow + b`. What a GGUF holds, what a quantizer
 *   emits, what `toFloatArray()` and every `get()` assume.
 * - [INPUT_BLOCK_MAJOR] — flat block index `b * outputDim + o`: all output rows' blocks for one
 *   input block, contiguous. What every packed matmul kernel actually reads.
 *
 * The two coincide only when a row is a single block, which is why mixing them produced *plausible
 * but wrong numbers* rather than a crash (#968, #971). Carrying the order on the [Layout] is what
 * makes it knowable from the value instead of guessed from the type: a kernel declares the order it
 * reads, and the dispatcher inserts a visible relayout when the two disagree.
 */
public enum class BlockOrder {
    /** `o * blocksPerRow + b` — canonical, as stored in a file. */
    ROW_MAJOR,

    /** `b * outputDim + o` — the order the packed matmul kernels feed on. */
    INPUT_BLOCK_MAJOR,
}

/**
 * Strides, byte offset and contiguity of a view over a [Storage] (SKEEP-003 §0 *Layout*). Pure
 * metadata: it says *where* the elements of a shape live inside a byte range, never what they mean
 * (that is [Format]) and never who owns them (that is [Storage]).
 *
 * Strides are in **elements** of the layout's unit — for a dense format one element is
 * `format.dtype.sizeInBytes`; for a block-packed format ([blocked]) a stride step addresses one
 * *block*, which is what makes a packed weight sliceable and transposable without touching bytes
 * (rule 5: "a view is a TensorView with the same Storage, different Layout").
 *
 * @property shape the extents this layout addresses
 * @property strides one stride per dimension, in elements (or blocks for a blocked layout)
 * @property offsetElements element (or block) offset of the first element inside the storage
 * @property elementBytes bytes per element (dense) or per block (blocked)
 * @property blockAxis for a [blocked] layout, the axis whose extent is measured in blocks. It is
 *   the last axis as loaded, and it *moves* with [transpose]/[unsqueeze]/[squeeze] — which is what
 *   lets a packed weight be transposed as metadata and still decode correctly (#1034).
 * @property blockOrder for a [blocked] layout, the order its blocks are in ([BlockOrder]). Default
 *   [BlockOrder.ROW_MAJOR]: what a file holds. A buffer in kernel feed order says so here instead
 *   of leaving the next reader to guess (#973).
 */
public class Layout(
    public val shape: Shape,
    public val strides: IntArray,
    public val offsetElements: Long = 0L,
    public val elementBytes: Int = 4,
    public val blocked: Boolean = false,
    public val blockAxis: Int = shape.rank - 1,
    public val blockOrder: BlockOrder = BlockOrder.ROW_MAJOR,
) {
    init {
        require(strides.size == shape.rank) { "strides (${strides.size}) must match rank (${shape.rank})" }
        require(offsetElements >= 0) { "offsetElements must be >= 0" }
        require(elementBytes > 0) { "elementBytes must be > 0" }
        if (blocked) require(blockAxis in 0 until shape.rank) { "blockAxis $blockAxis out of range for rank ${shape.rank}" }
    }

    /** Number of elements addressed (the shape's volume). */
    public val elementCount: Long get() = shape.volume.toLong()

    /** Byte offset of the first element. */
    public val offsetBytes: Long get() = offsetElements * elementBytes

    /** Row-major (C-order) strides for [shape] — the canonical layout. */
    public val isRowMajor: Boolean get() = strides.contentEquals(rowMajorStrides(shape))

    /** Whether the elements occupy one gap-free run, i.e. the layout can be handed to a kernel as a flat range. */
    public val isContiguous: Boolean
        get() {
            if (shape.rank == 0) return true
            var expected = 1
            for (d in shape.rank - 1 downTo 0) {
                val extent = shape[d]
                if (extent == 1) continue          // a unit extent's stride is irrelevant
                if (strides[d] != expected) return false
                expected *= extent
            }
            return true
        }

    /** Flat element (or block) index of [indices] inside the storage, including [offsetElements]. */
    public fun indexOf(vararg indices: Int): Long {
        require(indices.size == shape.rank) { "expected ${shape.rank} indices, got ${indices.size}" }
        var flat = offsetElements
        for (d in indices.indices) {
            val i = indices[d]
            require(i in 0 until shape[d]) { "index $i out of range for axis $d (extent ${shape[d]})" }
            flat += i.toLong() * strides[d]
        }
        return flat
    }

    /** Byte offset of [indices]. */
    public fun byteOffsetOf(vararg indices: Int): Long = indexOf(*indices) * elementBytes

    /** A layout over `[from, to)` of [axis] — same strides, shifted offset (zero-copy slicing). */
    public fun narrow(axis: Int, from: Int, size: Int): Layout {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(from >= 0 && size >= 0 && from + size <= shape[axis]) { "narrow($axis, $from, $size) outside extent ${shape[axis]}" }
        val dims = shape.dimensions.copyOf(); dims[axis] = size
        return Layout(Shape(dims), strides.copyOf(), offsetElements + from.toLong() * strides[axis], elementBytes, blocked, blockAxis, blockOrder)
    }

    /** Swap two axes — metadata only; the bytes are untouched (the packed-transpose trick, rule 5). */
    public fun transpose(axis0: Int = shape.rank - 2, axis1: Int = shape.rank - 1): Layout {
        require(shape.rank >= 2) { "transpose needs rank >= 2" }
        require(axis0 in 0 until shape.rank && axis1 in 0 until shape.rank) { "axes out of range" }
        val dims = shape.dimensions.copyOf(); val st = strides.copyOf()
        val d = dims[axis0]; dims[axis0] = dims[axis1]; dims[axis1] = d
        val s = st[axis0]; st[axis0] = st[axis1]; st[axis1] = s
        // The block axis travels with its extent: a transposed packed view still knows which index
        // splits into (block, offset-within-block).
        val movedBlockAxis = when (blockAxis) { axis0 -> axis1; axis1 -> axis0; else -> blockAxis }
        return Layout(Shape(dims), st, offsetElements, elementBytes, blocked, movedBlockAxis, blockOrder)
    }

    /**
     * Every [step]-th element along [axis] — a stride multiply, zero-copy. This is what makes the
     * strided half of the old `Slice.Step` a layout operation rather than an index remapper
     * (#1034): the bytes are untouched and the result is an ordinary [Layout].
     */
    public fun step(axis: Int, step: Int): Layout {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(step > 0) { "step must be positive, got $step" }
        if (step == 1) return this
        val dims = shape.dimensions.copyOf()
        dims[axis] = (shape[axis] + step - 1) / step
        val st = strides.copyOf()
        st[axis] = strides[axis] * step
        return Layout(Shape(dims), st, offsetElements, elementBytes, blocked, blockAxis, blockOrder)
    }

    /** Insert a unit axis at [axis] (stride 0 — it is never stepped). */
    public fun unsqueeze(axis: Int): Layout {
        require(axis in 0..shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        val dims = IntArray(shape.rank + 1); val st = IntArray(shape.rank + 1)
        var j = 0
        for (i in 0..shape.rank) {
            if (i == axis) { dims[i] = 1; st[i] = 0 } else { dims[i] = shape[j]; st[i] = strides[j]; j++ }
        }
        return Layout(Shape(dims), st, offsetElements, elementBytes, blocked, if (axis <= blockAxis) blockAxis + 1 else blockAxis, blockOrder)
    }

    /** Drop the unit axis at [axis]. */
    public fun squeeze(axis: Int): Layout {
        require(axis in 0 until shape.rank) { "axis $axis out of range for rank ${shape.rank}" }
        require(shape[axis] == 1) { "axis $axis has extent ${shape[axis]}, not 1" }
        val dims = ArrayList<Int>(shape.rank - 1); val st = ArrayList<Int>(shape.rank - 1)
        require(!(blocked && axis == blockAxis)) { "cannot drop the block axis of a blocked layout" }
        for (i in 0 until shape.rank) if (i != axis) { dims += shape[i]; st += strides[i] }
        return Layout(Shape(dims.toIntArray()), st.toIntArray(), offsetElements, elementBytes, blocked, if (axis < blockAxis) blockAxis - 1 else blockAxis, blockOrder)
    }

    override fun toString(): String =
        "Layout($shape, strides=${strides.joinToString(",", "[", "]")}, offset=$offsetElements" +
            (if (blocked) " blocks/${blockOrder.name.lowercase()}" else "") +
            ", ${if (isContiguous) "contiguous" else "strided"})"

    override fun equals(other: Any?): Boolean = other is Layout && other.shape == shape &&
        other.strides.contentEquals(strides) && other.offsetElements == offsetElements &&
        other.elementBytes == elementBytes && other.blocked == blocked && other.blockAxis == blockAxis &&
        other.blockOrder == blockOrder

    override fun hashCode(): Int {
        var h = shape.hashCode()
        h = 31 * h + strides.contentHashCode(); h = 31 * h + offsetElements.hashCode()
        h = 31 * h + elementBytes; h = 31 * h + blocked.hashCode(); h = 31 * h + blockAxis
        h = 31 * h + blockOrder.hashCode()
        return h
    }

    public companion object {
        /** Row-major strides of [shape], in elements. */
        public fun rowMajorStrides(shape: Shape): IntArray {
            val st = IntArray(shape.rank)
            var acc = 1
            for (d in shape.rank - 1 downTo 0) { st[d] = acc; acc *= shape[d] }
            return st
        }

        /** The canonical dense row-major layout of [shape] for [format]. */
        public fun rowMajor(shape: Shape, format: Format, offsetElements: Long = 0L): Layout =
            Layout(shape, rowMajorStrides(shape), offsetElements, format.dtype.sizeInBytes, blocked = false)

        /**
         * A blocked layout: the last axis is measured in blocks of `blockSize` elements, one
         * "element" of the layout being one packed block of `bytesPerBlock` bytes. Used for the
         * GGML block formats, where a view addresses whole blocks (rule 5).
         */
        public fun blocked(
            shape: Shape,
            blockSize: Int,
            bytesPerBlock: Int,
            offsetBlocks: Long = 0L,
            blockOrder: BlockOrder = BlockOrder.ROW_MAJOR,
        ): Layout {
            require(blockSize > 0 && bytesPerBlock > 0) { "block geometry must be positive" }
            require(shape.rank >= 1) { "blocked layout needs rank >= 1" }
            val last = shape[shape.rank - 1]
            if (last % blockSize == 0) {
                val dims = shape.dimensions.copyOf()
                dims[dims.size - 1] = last / blockSize
                val blockShape = Shape(dims)
                // The order *is* a stride pattern: input-block-major means "step one row to move
                // one block index, step `rows` to move to the next input block". Expressing it in
                // the strides rather than in a branch keeps every other operation — narrow,
                // transpose, get — working unchanged on a prepacked view (#973).
                val strides = if (blockOrder == BlockOrder.INPUT_BLOCK_MAJOR && blockShape.rank == 2) {
                    intArrayOf(1, blockShape[0])
                } else {
                    rowMajorStrides(blockShape)
                }
                return Layout(blockShape, strides, offsetBlocks, bytesPerBlock, blocked = true, blockOrder = blockOrder)
            }
            // A block that spans rows (e.g. ternary, where the whole tensor is one block): address the
            // flattened element sequence instead. Such a view decodes but cannot be sliced per axis.
            require(shape.volume % blockSize == 0) {
                "neither the last extent ($last) nor the volume (${shape.volume}) is a multiple of the block size $blockSize"
            }
            val flat = Shape(shape.volume / blockSize)
            return Layout(flat, rowMajorStrides(flat), offsetBlocks, bytesPerBlock, blocked = true, blockOrder = blockOrder)
        }
    }
}
