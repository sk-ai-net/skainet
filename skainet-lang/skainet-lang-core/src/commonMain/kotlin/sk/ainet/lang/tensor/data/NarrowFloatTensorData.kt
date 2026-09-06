package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.NarrowFloatCodec

/**
 * Tensor data whose elements are stored as **packed 16-bit floats** — two bytes per element,
 * little-endian, no block structure and no per-block scale. The encoding is `Dense(2)`.
 *
 * This is the recognition surface for narrow-float dispatch: a matmul kernel can test
 * `is NarrowFloatTensorData` and read [packedData] directly at 2 bytes per element instead of
 * forcing a dequant-to-FP32 copy first. [codec] says which 16-bit format the bytes are in, so one
 * kernel implementation serves both BF16 and FP16.
 *
 * `get`/`copyToFloatArray` decode to FP32, so consumers that do not care about the storage width
 * see an ordinary float tensor. `set` re-encodes and is lossy by construction.
 *
 * @see sk.ainet.lang.types.Bf16Codec
 * @see sk.ainet.lang.types.Fp16Codec
 */
public interface NarrowFloatTensorData : TensorData<DType, Float> {

    /**
     * Raw packed bytes — 2 per logical element, little-endian. Length is `shape.volume * 2`.
     * Safe for direct hand-off to native / Panama kernels without an intermediate copy.
     */
    public val packedData: ByteArray

    /** The 16-bit format [packedData] is encoded in. */
    public val codec: NarrowFloatCodec

    public companion object {
        /** Bytes per element for every narrow float format. */
        public const val BYTES_PER_ELEMENT: Int = 2
    }
}

/**
 * Dense narrow-float tensor data backed by a packed byte array.
 *
 * Memory layout: row-major; element at flat index `i` occupies bytes `[i*2 .. i*2+1]`, low byte
 * first. The concrete 16-bit format is supplied as a [NarrowFloatCodec], which is what lets BF16
 * and FP16 share one implementation rather than maintaining two parallel stacks.
 *
 * @param initialShape logical shape, in elements (not bytes).
 * @param data packed bytes, length ≥ `shape.volume * 2`.
 * @param codec the 16-bit format of [data].
 */
public open class NarrowFloatDenseTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    override val codec: NarrowFloatCodec,
) : NarrowFloatTensorData {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    /** Physically two bytes per element whatever the declared dtype witness. */
    override val encoding: TensorEncoding get() = TensorEncoding.Dense(NarrowFloatTensorData.BYTES_PER_ELEMENT)

    /**
     * A view over the *same* packed bytes, decoded by this data's [codec] (SKEEP-003 §4.1 façade).
     * The dtype is the codec's (FP16 or BF16) and the encoding `Dense(2)`; `view.get()` returns the
     * decoded float, exactly like [get].
     */
    override val view: sk.ainet.lang.memory.TensorView
        get() = sk.ainet.lang.memory.TensorView(
            shape = shape,
            format = sk.ainet.lang.memory.Format(codec.dtype, TensorEncoding.Dense(NarrowFloatTensorData.BYTES_PER_ELEMENT)),
            layout = sk.ainet.lang.memory.Layout(
                shape = shape,
                strides = sk.ainet.lang.memory.Layout.rowMajorStrides(shape),
                elementBytes = NarrowFloatTensorData.BYTES_PER_ELEMENT,
            ),
            storage = sk.ainet.lang.memory.Storage.Heap.wrap(data, mutable = false),
            decoder = sk.ainet.lang.memory.NarrowFloatDecoder(codec),
        )

    init {
        val requiredBytes = shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes " +
                "for ${shape.volume} ${codec.dtype.name} elements"
        }
    }

    override fun get(vararg indices: Int): Float {
        val byteIdx = calcFlatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val lo = data[byteIdx].toInt() and 0xFF
        val hi = data[byteIdx + 1].toInt() and 0xFF
        return codec.decode((hi shl 8) or lo)
    }

    override fun set(vararg indices: Int, value: Float) {
        val byteIdx = calcFlatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val bits = codec.encode(value)
        data[byteIdx] = (bits and 0xFF).toByte()
        data[byteIdx + 1] = ((bits ushr 8) and 0xFF).toByte()
    }

    override fun copyToFloatArray(): FloatArray {
        val volume = shape.volume
        val out = FloatArray(volume)
        for (i in 0 until volume) {
            val byteIdx = i * NarrowFloatTensorData.BYTES_PER_ELEMENT
            val lo = data[byteIdx].toInt() and 0xFF
            val hi = data[byteIdx + 1].toInt() and 0xFF
            out[i] = codec.decode((hi shl 8) or lo)
        }
        return out
    }

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }

    public companion object {
        /** Pack a `FloatArray` into [codec]'s 16-bit format. Lossy by construction. */
        public fun fromFloatArray(
            shape: Shape,
            values: FloatArray,
            codec: NarrowFloatCodec,
        ): NarrowFloatDenseTensorData {
            require(values.size >= shape.volume) {
                "FloatArray length ${values.size} is less than ${shape.volume} elements required"
            }
            val bytes = ByteArray(shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT)
            for (i in 0 until shape.volume) {
                val bits = codec.encode(values[i])
                bytes[i * 2] = (bits and 0xFF).toByte()
                bytes[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
            }
            return NarrowFloatDenseTensorData(shape, bytes, codec)
        }
    }
}

/**
 * Dense **IEEE binary16** tensor data.
 *
 * The counterpart to [Bf16DenseTensorData], and the class the SafeTensors and GGUF loaders name in
 * their "no FP16 backing yet" errors — its absence was the reason `Require(FP16)` had to be
 * rejected outright.
 */
public class Fp16DenseTensorData(
    initialShape: Shape,
    data: ByteArray,
) : NarrowFloatDenseTensorData(initialShape, data, Fp16Codec) {

    public companion object {
        /** Wrap raw packed FP16 bytes; length must be ≥ `shape.volume * 2`. */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Fp16DenseTensorData =
            Fp16DenseTensorData(shape, bytes)

        /** Build from FP32 values, rounding each to nearest binary16 (ties to even). */
        public fun fromFloatArray(shape: Shape, values: FloatArray): Fp16DenseTensorData {
            require(values.size >= shape.volume) {
                "FloatArray length ${values.size} is less than ${shape.volume} FP16 elements required"
            }
            val bytes = ByteArray(shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT)
            for (i in 0 until shape.volume) {
                val bits = Fp16Codec.encode(values[i])
                bytes[i * 2] = (bits and 0xFF).toByte()
                bytes[i * 2 + 1] = ((bits ushr 8) and 0xFF).toByte()
            }
            return Fp16DenseTensorData(shape, bytes)
        }
    }
}

/**
 * A rank-2 narrow-float weight whose bytes are stored **input-major**: the element at logical
 * `[row, col]` sits at flat byte index `(col * rows + row) * 2`, the transpose of the usual
 * row-major order.
 *
 * ### Why this exists
 *
 * Projections are stored `[out, in]` on disk, but `chooseQuantizedMatmul` needs `[in, out]`, so
 * every `Linear.onForward` calls `weight.t()` first. Transposing a row-major narrow tensor has no
 * fast path — it walks the tensor elementwise through boxed `get()` and widens to FP32, which at
 * real projection sizes costs hundreds of milliseconds to seconds *per weight, per token*. That
 * made KEEP_NATIVE slower than not using it at all.
 *
 * Storing the bytes input-major once, at load, makes that transpose free: input-major storage of
 * `[out, in]` **is** row-major storage of `[in, out]`, so [transposedView] hands back an ordinary
 * [NarrowFloatDenseTensorData] over the very same buffer. No copy, and — unlike the lazy transpose
 * used for the K-quants — element access stays correct on both sides, because each type indexes
 * the shared bytes with the strides its own shape implies.
 *
 * The layout is carried in the type rather than a flag so that a bare shape swap cannot be applied
 * to a row-major buffer by accident. Doing that would not throw; it would silently reinterpret the
 * weight as a different matrix.
 *
 * Build these with [fromRowMajor], which performs the one-off relayout.
 */
public class NarrowFloatInputMajorTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    override val codec: NarrowFloatCodec,
) : NarrowFloatTensorData {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    override val packedData: ByteArray get() = data

    private val rows: Int = shape.dimensions[0]
    private val cols: Int = shape.dimensions[1]

    init {
        require(shape.dimensions.size == 2) {
            "Input-major layout is only defined for rank-2 weights, got shape $shape"
        }
        val requiredBytes = shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes " +
                "for ${shape.volume} ${codec.dtype.name} elements"
        }
    }

    /** Flat index of logical `[row, col]` under input-major storage. */
    private fun flatIndex(indices: IntArray): Int {
        require(indices.size == 2) {
            "Number of indices (${indices.size}) must match tensor dimensions (2)"
        }
        val row = indices[0]
        val col = indices[1]
        require(row in 0 until rows) { "Index $row out of bounds for dimension 0 with size $rows" }
        require(col in 0 until cols) { "Index $col out of bounds for dimension 1 with size $cols" }
        return col * rows + row
    }

    override fun get(vararg indices: Int): Float {
        val byteIdx = flatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val lo = data[byteIdx].toInt() and 0xFF
        val hi = data[byteIdx + 1].toInt() and 0xFF
        return codec.decode((hi shl 8) or lo)
    }

    override fun set(vararg indices: Int, value: Float) {
        val byteIdx = flatIndex(indices) * NarrowFloatTensorData.BYTES_PER_ELEMENT
        val bits = codec.encode(value)
        data[byteIdx] = (bits and 0xFF).toByte()
        data[byteIdx + 1] = ((bits ushr 8) and 0xFF).toByte()
    }

    /** Decodes in logical row-major order, so callers see the same values a dense tensor would. */
    override fun copyToFloatArray(): FloatArray {
        val out = FloatArray(shape.volume)
        var dst = 0
        for (row in 0 until rows) {
            for (col in 0 until cols) {
                val byteIdx = (col * rows + row) * NarrowFloatTensorData.BYTES_PER_ELEMENT
                val lo = data[byteIdx].toInt() and 0xFF
                val hi = data[byteIdx + 1].toInt() and 0xFF
                out[dst++] = codec.decode((hi shl 8) or lo)
            }
        }
        return out
    }

    /**
     * The transpose, sharing this instance's buffer — no copy. Input-major `[rows, cols]` is
     * row-major `[cols, rows]`, so the result is an ordinary dense narrow tensor that
     * `chooseQuantizedMatmul` accepts directly.
     */
    public fun transposedView(): NarrowFloatDenseTensorData =
        NarrowFloatDenseTensorData(Shape(cols, rows), data, codec)

    public companion object {
        /**
         * Relayout `rowMajorBytes` (logical `[rows, cols]`, row-major) into input-major order.
         * This is the one-off cost that buys a free transpose on every later forward pass.
         */
        public fun fromRowMajor(
            shape: Shape,
            rowMajorBytes: ByteArray,
            codec: NarrowFloatCodec,
        ): NarrowFloatInputMajorTensorData {
            require(shape.dimensions.size == 2) {
                "Input-major layout is only defined for rank-2 weights, got shape $shape"
            }
            val rows = shape.dimensions[0]
            val cols = shape.dimensions[1]
            val required = shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT
            require(rowMajorBytes.size >= required) {
                "Data size ${rowMajorBytes.size} is less than required $required bytes " +
                    "for ${shape.volume} ${codec.dtype.name} elements"
            }
            val out = ByteArray(required)
            for (row in 0 until rows) {
                for (col in 0 until cols) {
                    val src = (row * cols + col) * NarrowFloatTensorData.BYTES_PER_ELEMENT
                    val dst = (col * rows + row) * NarrowFloatTensorData.BYTES_PER_ELEMENT
                    out[dst] = rowMajorBytes[src]
                    out[dst + 1] = rowMajorBytes[src + 1]
                }
            }
            return NarrowFloatInputMajorTensorData(shape, out, codec)
        }
    }
}

/** Decode a [NarrowFloatTensorData] to a fresh FloatArray. */
public fun NarrowFloatTensorData.toFloatArray(): FloatArray = copyToFloatArray()
