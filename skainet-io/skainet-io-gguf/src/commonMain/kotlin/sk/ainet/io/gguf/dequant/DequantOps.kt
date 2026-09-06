package sk.ainet.io.gguf.dequant

import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.QK_K
import kotlin.math.pow

/**
 * Shared dequantization operations for GGUF and SafeTensors weight loading.
 *
 * All functions are module-visible (`internal`) and stateless. They convert
 * quantized byte payloads into FP32 float arrays.
 */
public object DequantOps {

    // ========== List<Any>-based variants (GGUFReader in-memory) ==========

    @OptIn(ExperimentalUnsignedTypes::class)
    public fun toByteArray(raw: List<Any>, tensorName: String): ByteArray {
        val first = raw.firstOrNull()
        return when (first) {
            is Byte -> ByteArray(raw.size) { (raw[it] as Number).toByte() }
            is UByte -> ByteArray(raw.size) { (raw[it] as UByte).toByte() }
            else -> error("Unexpected raw data type ${typeName(first)} for tensor $tensorName")
        }
    }

    public fun dequantF16(raw: List<Any>): FloatArray {
        val bytes: ByteArray = toByteArray(raw, "F16")
        return dequantF16FromBytes(bytes)
    }

    public fun dequantBF16(raw: List<Any>): FloatArray {
        val bytes: ByteArray = toByteArray(raw, "BF16")
        return dequantBF16FromBytes(bytes)
    }

    public fun dequantQ4_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_0")
        return dequantQ4_0FromBytes(bytes, nElems)
    }

    public fun dequantQ5_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_0")
        return dequantQ5_0FromBytes(bytes, nElems)
    }

    public fun dequantQ8_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_0")
        return dequantQ8_0FromBytes(bytes, nElems)
    }

    public fun dequantQ4_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_1")
        return dequantQ4_1FromBytes(bytes, nElems)
    }

    public fun dequantQ5_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_1")
        return dequantQ5_1FromBytes(bytes, nElems)
    }

    public fun dequantQ8_1(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_1")
        return dequantQ8_1FromBytes(bytes, nElems)
    }

    public fun dequantIQ4NL(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ4_NL")
        return dequantIQ4NLFromBytes(bytes, nElems)
    }

    public fun dequantIQ4XS(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "IQ4_XS")
        return dequantIQ4XSFromBytes(bytes, nElems)
    }

    public fun dequantQ2K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q2_K")
        return dequantQ2KFromBytes(bytes, nElems)
    }

    public fun dequantQ3K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q3_K")
        return dequantQ3KFromBytes(bytes, nElems)
    }

    public fun dequantQ4K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q4_K")
        return dequantQ4KFromBytes(bytes, nElems)
    }

    public fun dequantQ5K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q5_K")
        return dequantQ5KFromBytes(bytes, nElems)
    }

    public fun dequantQ6K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q6_K")
        return dequantQ6KFromBytes(bytes, nElems)
    }

    public fun dequantQ8K(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "Q8_K")
        return dequantQ8KFromBytes(bytes, nElems)
    }

    public fun dequantTQ2_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "TQ2_0")
        return dequantTQ2_0FromBytes(bytes, nElems)
    }

    public fun dequantTQ1_0(raw: List<Any>, nElems: Int): FloatArray {
        val bytes = toByteArray(raw, "TQ1_0")
        return dequantTQ1_0FromBytes(bytes, nElems)
    }

    // ========== ByteArray-based variants (StreamingGGUFReader / SafeTensors) ==========

    public fun dequantF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            val half = (b1 shl 8) or b0
            out[o] = halfToFloat(half)
            i += 2
            o++
        }
        return out
    }

    public fun dequantBF16FromBytes(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val b0 = bytes[i].toInt() and 0xFF
            val b1 = bytes[i + 1].toInt() and 0xFF
            val bits = (b1 shl 24) or (b0 shl 16)
            out[o] = Float.fromBits(bits)
            i += 2
            o++
        }
        return out
    }

    public fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 4)
        var i = 0
        var o = 0
        while (i < bytes.size) {
            val bits = (bytes[i].toInt() and 0xFF) or
                ((bytes[i + 1].toInt() and 0xFF) shl 8) or
                ((bytes[i + 2].toInt() and 0xFF) shl 16) or
                ((bytes[i + 3].toInt() and 0xFF) shl 24)
            out[o] = Float.fromBits(bits)
            i += 4
            o++
        }
        return out
    }

    public fun halfToFloat(hbits: Int): Float {
        val mant = hbits and 0x03FF
        val exp = hbits and 0x7C00
        val sign = hbits and 0x8000
        return when (exp) {
            0 -> {
                val v = (mant.toFloat() / 1024.0f) * (2.0f).pow(-14)
                if (sign != 0) -v else v
            }
            0x7C00 -> {
                val v = if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                if (sign != 0) -v else v
            }
            else -> {
                val v = (1.0f + mant.toFloat() / 1024.0f) * (2.0f).pow((exp shr 10) - 15)
                if (sign != 0) -v else v
            }
        }
    }

    /**
     * Handle column-major to row-major conversion for GGUF tensors.
     * Data layout is unchanged — only the shape interpretation changes.
     */
    @Suppress("UNUSED_PARAMETER")
    public fun transposeColumnMajorToRowMajor(
        data: FloatArray,
        rows: Int,
        cols: Int
    ): FloatArray = data

    /**
     * Dispatch dequantization based on tensor type for byte arrays.
     */
    fun dequantFromBytes(bytes: ByteArray, tensorType: GGMLQuantizationType, nElems: Int): FloatArray {
        return when (tensorType) {
            GGMLQuantizationType.F16 -> dequantF16FromBytes(bytes)
            GGMLQuantizationType.BF16 -> dequantBF16FromBytes(bytes)
            GGMLQuantizationType.Q4_0 -> dequantQ4_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q4_1 -> dequantQ4_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_0 -> dequantQ5_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_1 -> dequantQ5_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_0 -> dequantQ8_0FromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_1 -> dequantQ8_1FromBytes(bytes, nElems)
            GGMLQuantizationType.Q2_K -> dequantQ2KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q3_K -> dequantQ3KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q4_K -> dequantQ4KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q5_K -> dequantQ5KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q6_K -> dequantQ6KFromBytes(bytes, nElems)
            GGMLQuantizationType.Q8_K -> dequantQ8KFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ4_NL -> dequantIQ4NLFromBytes(bytes, nElems)
            GGMLQuantizationType.IQ4_XS -> dequantIQ4XSFromBytes(bytes, nElems)
            GGMLQuantizationType.TQ1_0 -> dequantTQ1_0FromBytes(bytes, nElems)
            GGMLQuantizationType.TQ2_0 -> dequantTQ2_0FromBytes(bytes, nElems)
            else -> error("Dequantization for $tensorType not implemented")
        }
    }

    /**
     * Dispatch dequantization based on tensor type for List<Any> (in-memory GGUFReader).
     */
    public fun dequantFromList(raw: List<Any>, tensorType: GGMLQuantizationType, nElems: Int): FloatArray {
        return when (tensorType) {
            GGMLQuantizationType.F16 -> dequantF16(raw)
            GGMLQuantizationType.BF16 -> dequantBF16(raw)
            GGMLQuantizationType.Q4_0 -> dequantQ4_0(raw, nElems)
            GGMLQuantizationType.Q4_1 -> dequantQ4_1(raw, nElems)
            GGMLQuantizationType.Q5_0 -> dequantQ5_0(raw, nElems)
            GGMLQuantizationType.Q5_1 -> dequantQ5_1(raw, nElems)
            GGMLQuantizationType.Q8_0 -> dequantQ8_0(raw, nElems)
            GGMLQuantizationType.Q8_1 -> dequantQ8_1(raw, nElems)
            GGMLQuantizationType.Q2_K -> dequantQ2K(raw, nElems)
            GGMLQuantizationType.Q3_K -> dequantQ3K(raw, nElems)
            GGMLQuantizationType.Q4_K -> dequantQ4K(raw, nElems)
            GGMLQuantizationType.Q5_K -> dequantQ5K(raw, nElems)
            GGMLQuantizationType.Q6_K -> dequantQ6K(raw, nElems)
            GGMLQuantizationType.Q8_K -> dequantQ8K(raw, nElems)
            GGMLQuantizationType.IQ4_NL -> dequantIQ4NL(raw, nElems)
            GGMLQuantizationType.IQ4_XS -> dequantIQ4XS(raw, nElems)
            GGMLQuantizationType.TQ1_0 -> dequantTQ1_0(raw, nElems)
            GGMLQuantizationType.TQ2_0 -> dequantTQ2_0(raw, nElems)
            else -> error("Dequantization for $tensorType not implemented")
        }
    }

    /**
     * Returns (bytesPerBlock, elemsPerBlock) for a given quantization type.
     * Useful for chunked dequantization on single-threaded platforms (WASM).
     */
    public fun blockInfoFor(type: GGMLQuantizationType): Pair<Int, Int> = when (type) {
        GGMLQuantizationType.F16  -> Pair(2, 1)
        GGMLQuantizationType.BF16 -> Pair(2, 1)
        GGMLQuantizationType.F32  -> Pair(4, 1)
        GGMLQuantizationType.Q4_0 -> Pair(18, 32)
        GGMLQuantizationType.Q4_1 -> Pair(20, 32)
        GGMLQuantizationType.Q5_0 -> Pair(22, 32)
        GGMLQuantizationType.Q5_1 -> Pair(24, 32)
        GGMLQuantizationType.Q8_0 -> Pair(34, 32)
        GGMLQuantizationType.Q8_1 -> Pair(40, 32)
        GGMLQuantizationType.IQ4_NL -> Pair(18, 32)
        GGMLQuantizationType.IQ4_XS -> Pair(2 + 2 + QK_K / 2 + QK_K / 64, QK_K)
        GGMLQuantizationType.Q2_K -> Pair(2 + 2 + QK_K / 16 + QK_K / 4, QK_K)
        GGMLQuantizationType.Q3_K -> Pair(2 + QK_K / 4 + QK_K / 8 + 12, QK_K)
        GGMLQuantizationType.Q4_K -> Pair(144, QK_K)
        GGMLQuantizationType.Q5_K -> Pair(176, QK_K)
        GGMLQuantizationType.Q6_K -> Pair(210, QK_K)
        GGMLQuantizationType.Q8_K -> Pair(292, QK_K)
        GGMLQuantizationType.TQ1_0 -> Pair(54, 256)
        GGMLQuantizationType.TQ2_0 -> Pair(66, 256)
        else -> error("Block info for $type not available")
    }

    // ========== ByteArray-based quantization implementations ==========

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 18
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val b0 = bytes[offset].toInt() and 0xFF
            val b1 = bytes[offset + 1].toInt() and 0xFF
            val d = halfToFloat(b0 or (b1 shl 8))
            offset += 2
            for (j in 0 until 16) {
                val b = bytes[offset + j].toInt() and 0xFF
                val lo = (b and 0x0F) - 8
                val hi = (b shr 4) - 8
                out[outOff + j] = lo.toFloat() * d
                out[outOff + 16 + j] = hi.toFloat() * d
            }
            offset += 16
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 22
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            offset += 2
            val qh0 = bytes[offset].toInt() and 0xFF
            val qh1 = bytes[offset + 1].toInt() and 0xFF
            val qh2 = bytes[offset + 2].toInt() and 0xFF
            val qh3 = bytes[offset + 3].toInt() and 0xFF
            offset += 4
            val qh = intArrayOf(qh0, qh1, qh2, qh3)
            for (j in 0 until 16) {
                val q = bytes[offset + j].toInt() and 0xFF
                val lo = q and 0x0F
                val hi = q shr 4
                val bitLo = ((qh[j / 8] shr (j % 8)) and 0x01) shl 4
                val bitHi = ((qh[(j + 16) / 8] shr ((j + 16) % 8)) and 0x01) shl 4
                out[outOff + j] = d * (lo + bitLo - 16)
                out[outOff + 16 + j] = d * (hi + bitHi - 16)
            }
            offset += 16
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 34
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            offset += 2
            for (j in 0 until 32) {
                out[outOff + j] = d * bytes[offset + j].toFloat()
            }
            offset += 32
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 20
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val m = halfToFloat((bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF))
            offset += 4
            for (j in 0 until 16) {
                val b = bytes[offset + j].toInt() and 0xFF
                val lo = b and 0x0F
                val hi = b shr 4
                out[outOff + j] = d * lo + m
                out[outOff + 16 + j] = d * hi + m
            }
            offset += 16
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 24
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            val m = halfToFloat((bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF))
            offset += 4
            val qh0 = bytes[offset].toInt() and 0xFF
            val qh1 = bytes[offset + 1].toInt() and 0xFF
            val qh2 = bytes[offset + 2].toInt() and 0xFF
            val qh3 = bytes[offset + 3].toInt() and 0xFF
            offset += 4
            val qh = intArrayOf(qh0, qh1, qh2, qh3)
            for (j in 0 until 16) {
                val q = bytes[offset + j].toInt() and 0xFF
                val lo = q and 0x0F
                val hi = q shr 4
                val bitLo = (qh[j / 8] shr (j % 8)) and 0x01
                val bitHi = (qh[(j + 16) / 8] shr ((j + 16) % 8)) and 0x01
                out[outOff + j] = d * (lo + (bitLo shl 4)) + m
                out[outOff + 16 + j] = d * (hi + (bitHi shl 4)) + m
            }
            offset += 16
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8_1FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 40
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val dBits = (bytes[offset].toInt() and 0xFF) or
                    ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[offset + 3].toInt() and 0xFF) shl 24)
            val d = Float.fromBits(dBits)
            offset += 8 // skip d (4 bytes) + s (4 bytes)
            for (j in 0 until 32) {
                out[outOff + j] = d * bytes[offset + j].toFloat()
            }
            offset += 32
            outOff += blockSize
        }
        return out
    }

    private val iq4nlValues: IntArray = intArrayOf(
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    )

    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ4NLFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = 32
        val bytesPerBlock = 18
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            offset += 2
            repeat(blockSize / 2) { j ->
                val code = bytes[offset + j].toInt() and 0xFF
                val lo = code and 0x0F
                val hi = code ushr 4
                out[outOff + j] = d * iq4nlValues[lo]
                out[outOff + blockSize / 2 + j] = d * iq4nlValues[hi]
            }
            offset += blockSize / 2
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantIQ4XSFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 2 + 2 + QK_K / 2 + QK_K / 64
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat((bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF))
            offset += 2
            val scalesH = (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            offset += 2
            val scalesL = bytes.copyOfRange(offset, offset + QK_K / 64)
            offset += QK_K / 64
            val qs = bytes.copyOfRange(offset, offset + QK_K / 2)
            offset += QK_K / 2
            repeat(QK_K / 32) { ib ->
                val ls = ((scalesL[ib / 2].toInt() ushr (4 * (ib % 2))) and 0x0F) or
                    (((scalesH ushr (2 * ib)) and 0x03) shl 4)
                val dl = d * (ls - 32)
                repeat(16) { j ->
                    val code = qs[ib * 16 + j].toInt() and 0xFF
                    val lo = code and 0x0F
                    val hi = code ushr 4
                    out[outOff + ib * 32 + j] = dl * iq4nlValues[lo]
                    out[outOff + ib * 32 + 16 + j] = dl * iq4nlValues[hi]
                }
            }
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ2KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 2 + 2 + QK_K / 16 + QK_K / 4
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            val dMin = halfToFloat(
                (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
            )
            offset += 4
            val scales = bytes.copyOfRange(offset, offset + 16)
            offset += 16
            val qs = bytes.copyOfRange(offset, offset + 64)
            offset += 64
            repeat(16) { block ->
                val scaleIdx = (scales[block].toInt() ushr 4) and 0x0F
                val minIdx = scales[block].toInt() and 0x0F
                val scale = d * (scaleIdx / 15.0f)
                val min = dMin * (minIdx / 15.0f)
                repeat(16) { j ->
                    val codeByte = qs[block * 4 + j / 4].toInt() and 0xFF
                    val q = (codeByte ushr ((j % 4) * 2)) and 0x03
                    out[outOff + block * 16 + j] = q * scale - min
                }
            }
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ3KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 2 + QK_K / 4 + QK_K / 8 + 12
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            offset += 2
            val hmask = bytes.copyOfRange(offset, offset + 32)
            offset += 32
            val qs = bytes.copyOfRange(offset, offset + 64)
            offset += 64
            val scales = bytes.copyOfRange(offset, offset + 12)
            offset += 12
            repeat(16) { block ->
                val bitPos = block * 6
                val bytePos = bitPos / 8
                val bitShift = bitPos % 8
                val packed =
                    (scales.getOrElse(bytePos) { 0 }.toInt() and 0xFF) or
                        ((scales.getOrElse(bytePos + 1) { 0 }.toInt() and 0xFF) shl 8) or
                        ((scales.getOrElse(bytePos + 2) { 0 }.toInt() and 0xFF) shl 16)
                val scaleIdx = (packed ushr bitShift) and 0x3F
                val scale = d * (scaleIdx / 63.0f)
                repeat(16) { j ->
                    val idx = block * 16 + j
                    val ql = (qs[idx / 4].toInt() ushr ((idx % 4) * 2)) and 0x03
                    val qh = (hmask[idx / 8].toInt() ushr (idx % 8)) and 0x01
                    val q = ql or (qh shl 2)
                    out[outOff + idx] = q * scale
                }
            }
            outOff += blockSize
        }
        return out
    }

    /**
     * ggml `get_scale_min_k4` reading straight from the packed buffer at
     * `scalesBase` — no per-block scratch copies (#782): the K-quant kernels
     * index into the source array directly so the only allocation a full-tensor
     * dequant makes is the destination `FloatArray`.
     */
    private fun getScaleMinK4(j: Int, bytes: ByteArray, scalesBase: Int): Pair<Int, Int> {
        return if (j < 4) {
            val sc = bytes[scalesBase + j].toInt() and 0x3F
            val m = bytes[scalesBase + j + 4].toInt() and 0x3F
            sc to m
        } else {
            val sc = ((bytes[scalesBase + j + 4].toInt() and 0x0F) or
                     (((bytes[scalesBase + j - 4].toInt() and 0xFF) shr 6) shl 4))
            val m = (((bytes[scalesBase + j + 4].toInt() and 0xFF) shr 4) or
                    (((bytes[scalesBase + j].toInt() and 0xFF) shr 6) shl 4))
            sc to m
        }
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ4KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 144
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            val dMin = halfToFloat(
                (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
            )
            offset += 4
            val scalesBase = offset
            offset += 12
            val qsBase = offset
            offset += 128

            var qOffset = 0
            var scaleIdx = 0
            repeat(4) {
                val (sc1, m1) = getScaleMinK4(scaleIdx, bytes, scalesBase)
                val (sc2, m2) = getScaleMinK4(scaleIdx + 1, bytes, scalesBase)
                val d1 = d * sc1
                val min1 = dMin * m1
                val d2 = d * sc2
                val min2 = dMin * m2

                for (l in 0 until 32) {
                    val q = bytes[qsBase + qOffset + l].toInt() and 0x0F
                    out[outOff++] = d1 * q - min1
                }
                for (l in 0 until 32) {
                    val q = (bytes[qsBase + qOffset + l].toInt() and 0xFF) shr 4
                    out[outOff++] = d2 * q - min2
                }
                qOffset += 32
                scaleIdx += 2
            }
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ5KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 176
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            val dMin = halfToFloat(
                (bytes[offset + 3].toInt() and 0xFF shl 8) or (bytes[offset + 2].toInt() and 0xFF)
            )
            offset += 4
            val scalesBase = offset
            offset += 12
            val qhBase = offset
            offset += 32
            val qsBase = offset
            offset += 128

            // Per ggml-quants.c `dequantize_row_q5_K`: the 32-byte qh is indexed
            // by `l` (0..31, same as qs's per-group byte position), and a single
            // bit is selected per (outer-iter, low/high nibble). Different
            // (outer, nibble) pairs use different bits of the SAME qh[l] byte:
            //   outer 0: low→bit0, hi→bit1
            //   outer 1: low→bit2, hi→bit3
            //   outer 2: low→bit4, hi→bit5
            //   outer 3: low→bit6, hi→bit7
            // (Earlier this code used `qh[idx/8]` indexed by output position,
            // which only happened to equal qh[l] for blockCount=1; on real
            // multi-block tensors like Gemma 4 E2B's per_layer_token_embd
            // (Q5_K, 1.6 GB) every 5th bit was wrong, corrupting the PLE
            // residual stream across all 35 layers.)
            var qOffset = 0
            var scaleIdx = 0
            var outIdx = 0
            for (outer in 0 until 4) {
                val (sc1, m1) = getScaleMinK4(scaleIdx, bytes, scalesBase)
                val (sc2, m2) = getScaleMinK4(scaleIdx + 1, bytes, scalesBase)
                val d1 = d * sc1
                val min1 = dMin * m1
                val d2 = d * sc2
                val min2 = dMin * m2
                val bitLow = 2 * outer
                val bitHi = 2 * outer + 1

                for (l in 0 until 32) {
                    val qLow = bytes[qsBase + qOffset + l].toInt() and 0x0F
                    val qHigh = ((bytes[qhBase + l].toInt() and 0xFF) ushr bitLow) and 0x01
                    val q = qLow or (qHigh shl 4)
                    out[outOff + outIdx + l] = d1 * q - min1
                }
                for (l in 0 until 32) {
                    val qLow = (bytes[qsBase + qOffset + l].toInt() and 0xFF) ushr 4
                    val qHigh = ((bytes[qhBase + l].toInt() and 0xFF) ushr bitHi) and 0x01
                    val q = qLow or (qHigh shl 4)
                    out[outOff + outIdx + 32 + l] = d2 * q - min2
                }
                qOffset += 32
                scaleIdx += 2
                outIdx += 64
            }
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ6KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 210
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val qlStart = offset
            offset += 128
            val qhStart = offset
            offset += 64
            val scalesStart = offset
            offset += 16
            val d = halfToFloat(
                (bytes[offset + 1].toInt() and 0xFF shl 8) or (bytes[offset].toInt() and 0xFF)
            )
            offset += 2

            repeat(2) { half ->
                val qlBase = qlStart + half * 64
                val qhBase = qhStart + half * 32
                val scBase = scalesStart + half * 8

                for (l in 0 until 32) {
                    val isIdx = l / 16

                    val q1Low = bytes[qlBase + l].toInt() and 0x0F
                    val q1High = (bytes[qhBase + l].toInt() shr 0) and 0x03
                    val q1 = (q1Low or (q1High shl 4)) - 32

                    val q2Low = bytes[qlBase + l + 32].toInt() and 0x0F
                    val q2High = (bytes[qhBase + l].toInt() shr 2) and 0x03
                    val q2 = (q2Low or (q2High shl 4)) - 32

                    val q3Low = (bytes[qlBase + l].toInt() and 0xFF) shr 4
                    val q3High = (bytes[qhBase + l].toInt() shr 4) and 0x03
                    val q3 = (q3Low or (q3High shl 4)) - 32

                    val q4Low = (bytes[qlBase + l + 32].toInt() and 0xFF) shr 4
                    val q4High = (bytes[qhBase + l].toInt() shr 6) and 0x03
                    val q4 = (q4Low or (q4High shl 4)) - 32

                    val sc1 = bytes[scBase + isIdx + 0].toInt()
                    val sc2 = bytes[scBase + isIdx + 2].toInt()
                    val sc3 = bytes[scBase + isIdx + 4].toInt()
                    val sc4 = bytes[scBase + isIdx + 6].toInt()

                    out[outOff + half * 128 + l + 0] = d * sc1 * q1
                    out[outOff + half * 128 + l + 32] = d * sc2 * q2
                    out[outOff + half * 128 + l + 64] = d * sc3 * q3
                    out[outOff + half * 128 + l + 96] = d * sc4 * q4
                }
            }
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    private fun dequantQ8KFromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blockSize = QK_K
        val bytesPerBlock = 292
        val blockCount = bytes.size / bytesPerBlock
        val out = FloatArray(blockCount * blockSize)
        var offset = 0
        var outOff = 0
        repeat(blockCount) {
            val dBits =
                (bytes[offset + 3].toInt() and 0xFF shl 24) or
                    (bytes[offset + 2].toInt() and 0xFF shl 16) or
                    (bytes[offset + 1].toInt() and 0xFF shl 8) or
                    (bytes[offset].toInt() and 0xFF)
            val d = Float.fromBits(dBits)
            offset += 4
            repeat(blockSize) { j ->
                out[outOff + j] = d * bytes[offset + j].toFloat()
            }
            offset += blockSize
            offset += 32 // skip bsums
            outOff += blockSize
        }
        return out
    }

    @Suppress("UNUSED_PARAMETER")
    /**
     * `TQ2_0` → floats through the shared reference codec (#1033).
     *
     * The layout — including the interleave, where one byte holds four elements **32 apart** — is
     * defined once by [sk.ainet.lang.memory.TernaryCodec], next to the `TensorEncoding.TQ2_0`
     * descriptor, so the loader and the ternary kernels cannot decode the same bytes differently.
     */
    private fun dequantTQ2_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blocks = bytes.size / sk.ainet.lang.tensor.storage.TensorEncoding.TQ2_0.BYTES_PER_BLOCK
        return sk.ainet.lang.memory.TernaryCodec.decodeTq2_0(bytes, blocks * 256)
    }

    /** `TQ1_0` → floats through the shared reference codec (#1033); see [dequantTQ2_0FromBytes]. */
    private fun dequantTQ1_0FromBytes(bytes: ByteArray, nElems: Int): FloatArray {
        val blocks = bytes.size / sk.ainet.lang.tensor.storage.TensorEncoding.TQ1_0.BYTES_PER_BLOCK
        return sk.ainet.lang.memory.TernaryCodec.decodeTq1_0(bytes, blocks * 256)
    }

    private fun typeName(value: Any?): String =
        value?.let { it::class.simpleName ?: it::class.toString() } ?: "null"
}
