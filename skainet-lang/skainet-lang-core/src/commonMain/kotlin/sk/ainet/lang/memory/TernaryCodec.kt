package sk.ainet.lang.memory

import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.Fp16Codec
import kotlin.math.abs
import kotlin.math.round

/**
 * Reference encoder/decoder for the ternary encodings, driven by their [BlockSpec].
 *
 * These are the *semantics* of `TQ1_0`, `TQ2_0` and BitNet b1.58 — plain, allocation-simple Kotlin
 * that runs on every target, the thing a NEON or SIMD kernel is later checked against (M2-F1/F2,
 * #1040/#1041). The GGML layouts are reproduced exactly as `ggml-quants.c`'s
 * `quantize_row_tq{1,2}_0_ref` / `dequantize_row_tq{1,2}_0` define them, **interleaving included**:
 *
 * - `TQ2_0`: byte `j + m` of a 32-byte chunk holds four elements 32 apart, element
 *   `chunk*128 + l*32 + m` in bit pair `l`.
 * - `TQ1_0`: five base-3 digits per byte, most significant first, the byte scaled by `256/243`;
 *   digits are read back with `(b * 3^n) mod 256 * 3 shr 8`. The first 32 bytes carry elements
 *   `n*32 + m`, the next 16 bytes `160 + n*16 + m`, the four `qh` bytes `240 + n*4 + j`.
 *
 * Both round-trip exactly for ternary input: `decode(encode(v)) == v * fp16(amax)`.
 */
public object TernaryCodec {

    /** Powers of three, as `dequantize_row_tq1_0`'s `pow3` table. */
    private val POW3 = intArrayOf(1, 3, 9, 27, 81, 243)

    private const val QK: Int = 256

    // --- TQ2_0 -------------------------------------------------------------------------------

    /**
     * Encode [values] (a whole number of 256-element blocks) to `TQ2_0` bytes, quantizing each
     * block by its own absmax as `quantize_row_tq2_0_ref` does.
     */
    public fun encodeTq2_0(values: FloatArray): ByteArray {
        val blocks = blockCountOf(values.size, TensorEncoding.TQ2_0)
        val out = ByteArray(blocks * TensorEncoding.TQ2_0.BYTES_PER_BLOCK)
        for (b in 0 until blocks) {
            val base = b * QK
            val d = absmax(values, base)
            val id = if (d != 0f) 1f / d else 0f
            val off = b * TensorEncoding.TQ2_0.BYTES_PER_BLOCK
            for (chunk in 0 until 2) {                       // two 32-byte chunks of 128 elements
                for (m in 0 until 32) {
                    var q = 0
                    for (n in 0 until 4) {
                        val xi = round(values[base + chunk * 128 + m + n * 32] * id).toInt() + 1
                        q = q or ((xi and 3) shl (2 * n))
                    }
                    out[off + chunk * 32 + m] = q.toByte()
                }
            }
            le16(out, off + 64, Fp16Codec.encode(d))
        }
        return out
    }

    /** Decode [elementCount] elements of `TQ2_0` [bytes] starting at [byteOffset]. */
    public fun decodeTq2_0(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): FloatArray {
        val codes = codesTq2_0(bytes, elementCount, byteOffset)
        return scaleByBlock(codes, bytes, byteOffset, TensorEncoding.TQ2_0.BYTES_PER_BLOCK, scaleOffset = 64)
    }

    /** The raw ternary codes (`-1, 0, +1`) of `TQ2_0` [bytes] — the form a ternary kernel wants. */
    public fun codesTq2_0(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): ByteArray {
        val out = ByteArray(elementCount)
        val blocks = (elementCount + QK - 1) / QK
        var w = 0
        for (b in 0 until blocks) {
            val off = byteOffset + b * TensorEncoding.TQ2_0.BYTES_PER_BLOCK
            for (chunk in 0 until 2) {
                for (l in 0 until 4) {
                    for (m in 0 until 32) {
                        if (w >= elementCount) return out
                        val q = (bytes[off + chunk * 32 + m].toInt() shr (l * 2)) and 3
                        out[w++] = (q - 1).toByte()
                    }
                }
            }
        }
        return out
    }

    // --- TQ1_0 -------------------------------------------------------------------------------

    /** Encode [values] (a whole number of 256-element blocks) to `TQ1_0` bytes. */
    public fun encodeTq1_0(values: FloatArray): ByteArray {
        val blocks = blockCountOf(values.size, TensorEncoding.TQ1_0)
        val out = ByteArray(blocks * TensorEncoding.TQ1_0.BYTES_PER_BLOCK)
        for (b in 0 until blocks) {
            val base = b * QK
            val d = absmax(values, base)
            val id = if (d != 0f) 1f / d else 0f
            val off = b * TensorEncoding.TQ1_0.BYTES_PER_BLOCK
            var e = base                                     // element cursor within the block
            for (m in 0 until 32) {                          // qs[0..31]: 5 digits, stride 32
                out[off + m] = base3Byte(values, e + m, 32, 5, id, extraTriple = false)
            }
            e += 5 * 32
            for (m in 0 until 16) {                          // qs[32..47]: 5 digits, stride 16
                out[off + 32 + m] = base3Byte(values, e + m, 16, 5, id, extraTriple = false)
            }
            e += 5 * 16
            for (j in 0 until 4) {                           // qh[0..3]: 4 digits, stride 4
                out[off + 48 + j] = base3Byte(values, e + j, 4, 4, id, extraTriple = true)
            }
            le16(out, off + 52, Fp16Codec.encode(d))
        }
        return out
    }

    /** Decode [elementCount] elements of `TQ1_0` [bytes] starting at [byteOffset]. */
    public fun decodeTq1_0(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): FloatArray {
        val codes = codesTq1_0(bytes, elementCount, byteOffset)
        return scaleByBlock(codes, bytes, byteOffset, TensorEncoding.TQ1_0.BYTES_PER_BLOCK, scaleOffset = 52)
    }

    /** The raw ternary codes (`-1, 0, +1`) of `TQ1_0` [bytes]. */
    public fun codesTq1_0(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): ByteArray {
        val out = ByteArray(elementCount)
        val blocks = (elementCount + QK - 1) / QK
        var w = 0
        for (b in 0 until blocks) {
            val off = byteOffset + b * TensorEncoding.TQ1_0.BYTES_PER_BLOCK
            for (n in 0 until 5) {                           // 32-byte section → 160 elements
                for (m in 0 until 32) {
                    if (w >= elementCount) return out
                    out[w++] = digit(bytes[off + m].toInt(), n)
                }
            }
            for (n in 0 until 5) {                           // 16-byte section → 80 elements
                for (m in 0 until 16) {
                    if (w >= elementCount) return out
                    out[w++] = digit(bytes[off + 32 + m].toInt(), n)
                }
            }
            for (n in 0 until 4) {                           // qh → the last 16 elements
                for (j in 0 until 4) {
                    if (w >= elementCount) return out
                    out[w++] = digit(bytes[off + 48 + j].toInt(), n)
                }
            }
        }
        return out
    }

    // --- BitNet b1.58 ------------------------------------------------------------------------

    /**
     * Encode [values] as BitNet b1.58: ternary codes four per byte in element order, followed by
     * one FP32 scale for the whole tensor (the absmean quantizer's `1/scale` rounding, clamped to
     * `-1, 0, +1`).
     */
    public fun encodeBitNet(values: FloatArray): ByteArray {
        val scale = absmean(values)
        val id = if (scale != 0f) 1f / scale else 0f
        val payload = (values.size + 3) / 4
        val out = ByteArray(payload + TensorEncoding.BITNET_B1_58.SCALE_BYTES)
        for (i in values.indices) {
            val code = round(values[i] * id).toInt().coerceIn(-1, 1) + 1
            val byteIndex = i / 4
            val shift = (i % 4) * 2
            out[byteIndex] = (out[byteIndex].toInt() or (code shl shift)).toByte()
        }
        le32(out, payload, scale.toRawBits())
        return out
    }

    /** Decode [elementCount] BitNet b1.58 elements: `code - 1` times the tensor's FP32 scale. */
    public fun decodeBitNet(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): FloatArray {
        val payload = (elementCount + 3) / 4
        val scale = Float.fromBits(le32(bytes, byteOffset + payload))
        val out = FloatArray(elementCount)
        for (i in 0 until elementCount) {
            val code = (bytes[byteOffset + i / 4].toInt() shr ((i % 4) * 2)) and 3
            out[i] = (code - 1).toFloat() * scale
        }
        return out
    }

    /** The FP32 scale of a BitNet b1.58 buffer holding [elementCount] elements. */
    public fun bitNetScale(bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): Float =
        Float.fromBits(le32(bytes, byteOffset + (elementCount + 3) / 4))

    // --- BitNet multi-plane trit residual (BITNET_PLANES, #1150) ------------------------------

    /**
     * Encode a `[rows, cols]` weight as [TensorEncoding.BITNET_PLANES] — the Kotlin port of
     * NeoGPU's `hs_mlt_lmhead_encode`: per row, `scale = max|row|` stored as FP16; the row is
     * normalized and decomposed by repeated round-to-trit (±0.5 threshold) with ×3 residual
     * scaling into [TensorEncoding.BITNET_PLANES.PLANES] planes, each sequentially 2-bit-packed.
     */
    public fun encodeBitNetPlanes(values: FloatArray, rows: Int, cols: Int): ByteArray {
        require(cols % 4 == 0) { "BITNET_PLANES needs cols % 4 == 0; got $cols" }
        require(values.size == rows * cols) { "values (${values.size}) must be rows*cols (${rows * cols})" }
        val planes = TensorEncoding.BITNET_PLANES.PLANES
        val planeStride = TensorEncoding.BITNET_PLANES.planeStrideBytes(rows, cols)
        val scalesOffset = TensorEncoding.BITNET_PLANES.rowScalesByteOffset(rows, cols)
        val out = ByteArray(TensorEncoding.BITNET_PLANES.bufferBytes(rows, cols))
        val rowBytes = cols / 4
        val residual = FloatArray(cols)
        for (r in 0 until rows) {
            var scale = 0f
            for (c in 0 until cols) {
                val a = abs(values[r * cols + c])
                if (a > scale) scale = a
            }
            val scaleBits = sk.ainet.lang.types.Fp16Codec.encode(scale)
            out[scalesOffset + r * 2] = (scaleBits and 0xFF).toByte()
            out[scalesOffset + r * 2 + 1] = ((scaleBits shr 8) and 0xFF).toByte()
            // Decompose against the FP16-rounded scale the decoder will read, not the exact one —
            // otherwise the stored trits answer for a scale that no longer exists.
            val storedScale = sk.ainet.lang.types.Fp16Codec.decode(scaleBits)
            val inv = if (storedScale != 0f) 1f / storedScale else 0f
            for (c in 0 until cols) residual[c] = values[r * cols + c] * inv
            for (p in 0 until planes) {
                val base = p * planeStride + r * rowBytes
                for (c in 0 until cols) {
                    val v = residual[c]
                    val t = if (v > 0.5f) 1 else if (v < -0.5f) -1 else 0
                    val shift = (c % 4) * 2
                    out[base + c / 4] = (out[base + c / 4].toInt() or ((t + 1) shl shift)).toByte()
                    residual[c] = (v - t) * 3f
                }
            }
        }
        return out
    }

    /** Decode a full [TensorEncoding.BITNET_PLANES] buffer back to `rows*cols` floats (all planes). */
    public fun decodeBitNetPlanes(bytes: ByteArray, rows: Int, cols: Int, byteOffset: Int = 0): FloatArray {
        val out = FloatArray(rows * cols)
        for (r in 0 until rows) decodeBitNetPlanesRow(bytes, rows, cols, r, out, r * cols, byteOffset)
        return out
    }

    /** Decode one row of a [TensorEncoding.BITNET_PLANES] buffer into [out] at [outOffset]. */
    public fun decodeBitNetPlanesRow(
        bytes: ByteArray,
        rows: Int,
        cols: Int,
        row: Int,
        out: FloatArray,
        outOffset: Int,
        byteOffset: Int = 0,
    ) {
        val planeStride = TensorEncoding.BITNET_PLANES.planeStrideBytes(rows, cols)
        val rowBytes = cols / 4
        val scale = planesRowScale(bytes, rows, cols, row, byteOffset)
        for (c in 0 until cols) {
            var acc = 0f
            var w = 1f
            for (p in 0 until TensorEncoding.BITNET_PLANES.PLANES) {
                val b = bytes[byteOffset + p * planeStride + row * rowBytes + c / 4].toInt() and 0xFF
                acc += (((b shr ((c % 4) * 2)) and 3) - 1) * w
                w /= 3f
            }
            out[outOffset + c] = acc * scale
        }
    }

    /** The FP16 per-row scale of row [row] in a `[rows, cols]` [TensorEncoding.BITNET_PLANES] buffer. */
    public fun planesRowScale(bytes: ByteArray, rows: Int, cols: Int, row: Int, byteOffset: Int = 0): Float {
        val offset = byteOffset + TensorEncoding.BITNET_PLANES.rowScalesByteOffset(rows, cols) + row * 2
        val bits = (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8)
        return sk.ainet.lang.types.Fp16Codec.decode(bits)
    }

    // --- dispatch ----------------------------------------------------------------------------

    /**
     * Decode [elementCount] elements of any ternary [encoding] — the entry point a kernel, a
     * fixture generator or [TernaryBlockDecoder] uses instead of switching on the encoding itself.
     *
     * @throws IllegalArgumentException if [encoding] is not one of the ternary encodings with bytes
     *   of their own ([TensorEncoding.TernaryPacked] keeps its scale outside the buffer — decode it
     *   through its `TensorData`).
     */
    public fun decode(encoding: TensorEncoding, bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): FloatArray =
        when (encoding) {
            TensorEncoding.TQ1_0 -> decodeTq1_0(bytes, elementCount, byteOffset)
            TensorEncoding.TQ2_0 -> decodeTq2_0(bytes, elementCount, byteOffset)
            TensorEncoding.BITNET_B1_58 -> decodeBitNet(bytes, elementCount, byteOffset)
            else -> throw IllegalArgumentException("$encoding is not a self-contained ternary encoding")
        }

    /**
     * The ternary codes (`-1, 0, +1`) of any self-contained ternary [encoding], scale not applied —
     * what a `bitnet_gemv` kernel consumes (#1040).
     */
    public fun codes(encoding: TensorEncoding, bytes: ByteArray, elementCount: Int, byteOffset: Int = 0): ByteArray =
        when (encoding) {
            TensorEncoding.TQ1_0 -> codesTq1_0(bytes, elementCount, byteOffset)
            TensorEncoding.TQ2_0 -> codesTq2_0(bytes, elementCount, byteOffset)
            TensorEncoding.BITNET_B1_58 -> ByteArray(elementCount) {
                (((bytes[byteOffset + it / 4].toInt() shr ((it % 4) * 2)) and 3) - 1).toByte()
            }
            else -> throw IllegalArgumentException("$encoding is not a self-contained ternary encoding")
        }

    /** Encode [values] in any ternary [encoding] — the inverse of [decode]. */
    public fun encode(encoding: TensorEncoding, values: FloatArray): ByteArray =
        when (encoding) {
            TensorEncoding.TQ1_0 -> encodeTq1_0(values)
            TensorEncoding.TQ2_0 -> encodeTq2_0(values)
            TensorEncoding.BITNET_B1_58 -> encodeBitNet(values)
            else -> throw IllegalArgumentException("$encoding is not a self-contained ternary encoding")
        }

    // --- helpers -----------------------------------------------------------------------------

    private fun blockCountOf(elementCount: Int, encoding: TensorEncoding): Int {
        val spec = encoding.blockSpec ?: throw IllegalArgumentException("$encoding has no block spec")
        require(elementCount % spec.blockSize == 0) {
            "${encoding.name} encodes whole blocks of ${spec.blockSize}; got $elementCount elements"
        }
        return elementCount / spec.blockSize
    }

    /** `q * 3^n mod 256`, then the leading base-3 digit — `dequantize_row_tq1_0`'s extraction. */
    private fun digit(byte: Int, n: Int): Byte {
        val q = ((byte and 0xFF) * POW3[n]) and 0xFF
        return (((q * 3) shr 8) - 1).toByte()
    }

    /** Multiply per-block codes by that block's FP16 scale, read at [scaleOffset] within the block. */
    private fun scaleByBlock(codes: ByteArray, bytes: ByteArray, byteOffset: Int, bytesPerBlock: Int, scaleOffset: Int): FloatArray {
        val out = FloatArray(codes.size)
        for (i in codes.indices) {
            val block = i / QK
            val d = Fp16Codec.decode(le16(bytes, byteOffset + block * bytesPerBlock + scaleOffset))
            out[i] = codes[i].toFloat() * d
        }
        return out
    }

    /**
     * Pack [digits] ternary values starting at [first], spaced [stride] apart, into one base-3 byte
     * scaled by `256/243` — `quantize_row_tq1_0_ref`. [extraTriple] applies the extra `*3` the
     * reference does for the four-digit `qh` bytes.
     */
    private fun base3Byte(values: FloatArray, first: Int, stride: Int, digits: Int, id: Float, extraTriple: Boolean): Byte {
        var q = 0
        for (n in 0 until digits) {
            val xi = round(values[first + n * stride] * id).toInt() + 1
            q = q * 3 + xi
        }
        if (extraTriple) q *= 3
        return ((q * 256 + 242) / 243).toByte()
    }

    private fun absmax(values: FloatArray, base: Int): Float {
        var amax = 0f
        for (j in 0 until QK) amax = maxOf(amax, abs(values[base + j]))
        return amax
    }

    private fun absmean(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var sum = 0.0
        for (v in values) sum += abs(v)
        return (sum / values.size).toFloat()
    }

    private fun le16(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte()
        b[off + 1] = ((v ushr 8) and 0xFF).toByte()
    }

    private fun le16(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8)

    private fun le32(b: ByteArray, off: Int, v: Int) {
        b[off] = (v and 0xFF).toByte()
        b[off + 1] = ((v ushr 8) and 0xFF).toByte()
        b[off + 2] = ((v ushr 16) and 0xFF).toByte()
        b[off + 3] = ((v ushr 24) and 0xFF).toByte()
    }

    private fun le32(b: ByteArray, off: Int): Int =
        (b[off].toInt() and 0xFF) or ((b[off + 1].toInt() and 0xFF) shl 8) or
            ((b[off + 2].toInt() and 0xFF) shl 16) or ((b[off + 3].toInt() and 0xFF) shl 24)
}

/**
 * A [BlockDecoder] for a ternary [encoding], reading its bytes straight out of the storage —
 * the descriptor-driven path a `TensorView` over `TQ1_0`/`TQ2_0`/BitNet bytes decodes through
 * (SKEEP-003 §4.4: `get()` decodes, never a raw byte).
 */
public class TernaryBlockDecoder private constructor(
    private val encoding: TensorEncoding,
    override val blockSize: Int,
    override val bytesPerBlock: Int,
) : BlockDecoder {

    /** A decoder for a block-structured ternary encoding ([TensorEncoding.TQ1_0], [TensorEncoding.TQ2_0]). */
    public constructor(encoding: TensorEncoding) : this(
        encoding,
        blockSize = requireBlocked(encoding).blockSize,
        bytesPerBlock = requireBlocked(encoding).bytesPerBlock,
    )

    /**
     * A decoder for a **per-tensor** ternary encoding ([TensorEncoding.BITNET_B1_58]), whose single
     * scale covers all [elementCount] elements: the whole tensor is one block (#1040).
     */
    public constructor(encoding: TensorEncoding, elementCount: Int) : this(
        encoding,
        blockSize = elementCount,
        bytesPerBlock = requirePerTensor(encoding, elementCount),
    )

    override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
        val heap = storage as? Storage.Heap
            ?: throw UnsupportedOperationException("ternary views need heap storage in this milestone")
        val bytes = heap.bytes ?: throw UnsupportedOperationException("ternary views need byte storage")
        val off = heap.arrayOffset + (blockIndex * bytesPerBlock).toInt()
        TernaryCodec.decode(encoding, bytes, blockSize, off).copyInto(out, outOffset)
    }

    private companion object {
        fun requireBlocked(encoding: TensorEncoding): BlockSpec {
            val spec = encoding.blockSpec ?: throw IllegalArgumentException("$encoding has no block spec")
            require(encoding.isTernary) { "$encoding is not a ternary encoding" }
            require(!spec.isPerTensor) {
                "${encoding.name} is a per-tensor encoding; give this constructor the element count"
            }
            return spec
        }

        fun requirePerTensor(encoding: TensorEncoding, elementCount: Int): Int {
            val spec = encoding.blockSpec ?: throw IllegalArgumentException("$encoding has no block spec")
            require(encoding.isTernary) { "$encoding is not a ternary encoding" }
            require(spec.isPerTensor) { "${encoding.name} is block-structured; use the single-argument constructor" }
            require(elementCount > 0) { "elementCount must be > 0" }
            return (encoding.physicalBytes(elementCount.toLong())
                ?: throw IllegalArgumentException("${encoding.name} cannot size $elementCount elements")).toInt()
        }
    }
}
