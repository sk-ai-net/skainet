package sk.ainet.lang.memory

import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.Int8
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * The int8 activations the ternary kernels consume (`W1.58A8`, SKEEP-003 §5.3, M2-F3) and the
 * adapter that produces them.
 *
 * A ternary weight is only cheap if the *other* operand is cheap too: `bitnet_gemv` adds and
 * subtracts activation values by the sign of the weight, so the activations are quantized to int8
 * with one absmax scale per token. This is exactly the "dispatcher-inserted adapter" of §5.1 — it
 * costs real bytes in the Forward scope every step, so it is allocated there and **traced**, never
 * hidden inside a kernel.
 *
 * Byte layout of a `[rows, cols]` activation: `rows * cols` codes, row-major, then `rows`
 * little-endian FP32 scales. `value ≈ code * scale(row)`.
 */
public object I8Absmax {

    /** The activation format: int8 values, per-token absmax scale. */
    public val FORMAT: Format = Format(Int8, TensorEncoding.DENSE_I8_ABSMAX)

    /** Bytes a `[rows, cols]` quantized activation occupies: the codes plus one scale per row. */
    public fun bytesFor(rows: Int, cols: Int): Long = rows.toLong() * cols + rows.toLong() * 4

    /**
     * Quantize [activation] (`[rows, cols]`, any decodable format) into [scope] as int8 codes with
     * a per-row absmax scale, and emit the adapter event that prices it.
     *
     * A row of zeros keeps scale `0` and decodes back to zeros rather than dividing by zero.
     */
    public fun requantize(
        activation: TensorView,
        scope: Scope,
        sink: TraceSink = NoopTraceSink,
        id: TensorId? = null,
    ): TensorView {
        require(activation.shape.rank == 2) { "activations are [rows, cols], got ${activation.shape}" }
        val rows = activation.shape[0]
        val cols = activation.shape[1]
        val bytes = ByteArray(bytesFor(rows, cols).toInt())
        val scaleBase = rows * cols
        for (r in 0 until rows) {
            var amax = 0f
            for (c in 0 until cols) amax = maxOf(amax, abs(activation.get(r, c)))
            val scale = amax / TensorEncoding.DENSE_I8_ABSMAX.CODE_RANGE
            val inverse = if (scale != 0f) 1f / scale else 0f
            for (c in 0 until cols) {
                val code = (activation.get(r, c) * inverse).roundToInt().coerceIn(-127, 127)
                bytes[r * cols + c] = code.toByte()
            }
            putFloat(bytes, scaleBase + r * 4, scale)
        }
        val storage = Storage.Heap.wrap(bytes, mutable = false, origin = id ?: activation.id)
        val target = id ?: activation.id
        if (sink.isEnabled) {
            sink.emit(
                TraceEvent.AdapterInserted(
                    kind = "requantize-i8-absmax",
                    from = activation.format,
                    to = FORMAT,
                    bytes = bytesFor(rows, cols),
                    target = target,
                    scope = scope.kind,
                ),
            )
        }
        return view(storage, rows, cols, target)
    }

    /** A view over already-quantized bytes: `rows * cols` codes then `rows` FP32 scales. */
    public fun view(storage: Storage, rows: Int, cols: Int, id: TensorId? = null): TensorView {
        val shape = Shape(rows, cols)
        return TensorView(
            shape = shape,
            format = FORMAT,
            layout = Layout(shape, Layout.rowMajorStrides(shape), 0L, 1),
            storage = storage,
            id = id,
            decoder = Decoder(rows, cols),
        )
    }

    /**
     * Decodes an int8 code back to `code * scale(row)` — so `view.get(r, c)` returns a *value*,
     * never a raw byte (rule 4), exactly like every other packed format.
     */
    private class Decoder(private val rows: Int, private val cols: Int) : BlockDecoder {
        override val blockSize: Int get() = 1
        override val bytesPerBlock: Int get() = 1

        override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
            out[outOffset] = decode(storage, blockIndex)
        }

        override fun decodeElement(storage: Storage, layout: Layout, flatElementIndex: Long): Float =
            decode(storage, flatElementIndex)

        private fun decode(storage: Storage, flatIndex: Long): Float {
            val heap = storage as? Storage.Heap
                ?: throw UnsupportedOperationException("I8-absmax activations live on the heap in this milestone")
            val bytes = heap.bytes ?: throw UnsupportedOperationException("I8-absmax activations need byte storage")
            val index = flatIndex.toInt()
            val row = index / cols
            return bytes[heap.arrayOffset + index] * scaleAt(bytes, heap.arrayOffset + rows * cols + row * 4)
        }
    }

    private fun scaleAt(bytes: ByteArray, offset: Int): Float = Float.fromBits(
        (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or ((bytes[offset + 3].toInt() and 0xFF) shl 24),
    )

    /** The int8 code at `(row, col)` of a view in [FORMAT]. */
    public fun codeAt(view: TensorView, row: Int, col: Int): Int {
        val bytes = bytesOf(view)
        return bytes[row * view.shape[1] + col].toInt()
    }

    /** The absmax scale of [row]. */
    public fun scaleOf(view: TensorView, row: Int): Float =
        scaleAt(bytesOf(view), view.shape[0] * view.shape[1] + row * 4)

    /** The decoded value at `(row, col)` — `code * scale(row)`. */
    public fun valueAt(view: TensorView, row: Int, col: Int): Float = codeAt(view, row, col) * scaleOf(view, row)

    /** The codes of [row] as a `ByteArray` view into the storage — what a kernel iterates. */
    public fun rowCodes(view: TensorView, row: Int): ByteArray {
        val bytes = bytesOf(view)
        val cols = view.shape[1]
        return bytes.copyOfRange(row * cols, row * cols + cols)
    }

    private fun bytesOf(view: TensorView): ByteArray {
        require(view.format == FORMAT) { "not an I8-absmax activation: ${view.format}" }
        val heap = view.storage as? Storage.Heap
            ?: throw UnsupportedOperationException("I8-absmax activations live on the heap in this milestone")
        return heap.bytes ?: throw UnsupportedOperationException("I8-absmax activations need byte storage")
    }

    private fun putFloat(bytes: ByteArray, offset: Int, value: Float) {
        val bits = value.toRawBits()
        bytes[offset] = (bits and 0xFF).toByte()
        bytes[offset + 1] = ((bits ushr 8) and 0xFF).toByte()
        bytes[offset + 2] = ((bits ushr 16) and 0xFF).toByte()
        bytes[offset + 3] = ((bits ushr 24) and 0xFF).toByte()
    }
}
