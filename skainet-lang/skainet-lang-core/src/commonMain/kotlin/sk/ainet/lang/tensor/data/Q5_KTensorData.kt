package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for Q5_K quantized format (canonical ggml layout).
 *
 * Q5_K block format (256 elements per block, 176 bytes per block):
 * - 2 bytes: f16 d (super-block scale)
 * - 2 bytes: f16 dMin (super-block min-scale)
 * - 12 bytes: packed 6-bit scaleIdx + 6-bit minIdx for each of 8 sub-blocks,
 *             encoded with ggml's `get_scale_min_k4` bit-mixing layout —
 *             *identical* to Q4_K (see [Q4_KTensorData]).
 * - 32 bytes: `qh` high-bit plane. One byte per intra-group element position
 *             `l` (0..31); the 5th bit of a code comes from a single bit of
 *             `qh[l]` selected by `(outer-iter, low/high nibble)`.
 * - 128 bytes: `qs` 4-bit low nibbles, laid out *strided* in 4 groups of 32
 *              bytes, exactly as Q4_K: byte (j*32 + i) carries element
 *              (2j*32 + i) in its lo nibble and element ((2j+1)*32 + i) in its
 *              hi nibble.
 *
 * The 5th bit (per ggml-quants.c `dequantize_row_q5_K`): for outer iteration
 * `outer` (0..3), the low-nibble sub-block uses `qh[l]` bit `2*outer` and the
 * high-nibble sub-block uses bit `2*outer + 1`. `qh` is indexed by the
 * intra-group position `l` (0..31), NOT by output position.
 *
 * Each sub-block s (s=0..7):
 * - 6-bit scaleIdx, 6-bit minIdx (from `get_scale_min_k4`)
 * - scale  = d    * scaleIdx
 * - offset = dMin * minIdx
 *
 * Dequantization: `output[i] = code[i] * scale - offset`, where `code` is the
 * full 5-bit value `lowNibble | (fifthBit << 4)` (0..31).
 *
 * Validated bit-exact against `DequantOps.dequantQ5KFromBytes`, which carries
 * the proof and the regression note about the earlier `qh[idx/8]` bug.
 */
public interface Q5_KTensorData : TensorData<DType, Byte> {
    /** Number of Q5_K blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    /** Get the main scale factor (d) for a block. */
    public fun getBlockD(blockIdx: Int): Float

    /** Get the minimum scale factor (dMin) for a block. */
    public fun getBlockDMin(blockIdx: Int): Float

    /**
     * Get the scale for a specific sub-block within a block:
     * `scale = d * scaleIdx` (no /63 normalisation — ggml's `d1 = d * sc`).
     */
    public fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Float

    /**
     * Get the offset for a specific sub-block within a block:
     * `offset = dMin * minIdx`. Subtract this from `code * scale` for the
     * dequantised value.
     */
    public fun getSubBlockMin(blockIdx: Int, subBlockIdx: Int): Float

    /** Get a 5-bit quantized code value (0..31) for `elementIdx` (0..255). */
    public fun getCode(blockIdx: Int, elementIdx: Int): Int

    public companion object {
        /** Elements per Q5_K block. */
        public const val BLOCK_SIZE: Int = 256

        /** Elements per sub-block. */
        public const val SUB_BLOCK_SIZE: Int = 32

        /** Number of sub-blocks per block. */
        public const val SUB_BLOCKS_PER_BLOCK: Int = 8

        /** Bytes per Q5_K block (2 + 2 + 12 + 32 + 128 = 176). */
        public const val BYTES_PER_BLOCK: Int = 176

        /** Byte offset of the 32-byte `qh` high-bit plane within a block. */
        public const val QH_OFFSET: Int = 16

        /** Byte offset of the 128-byte `qs` low-nibble region within a block. */
        public const val QS_OFFSET: Int = 48
    }
}

/**
 * Implementation of Q5_KTensorData backed by a packed byte array (canonical
 * ggml layout — see [Q5_KTensorData] kdoc for the full byte map).
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param data the raw packed block data
 */
public class Q5_KBlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q5_KTensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q5_KTensorData.BLOCK_SIZE - 1) / Q5_KTensorData.BLOCK_SIZE

    // PackedBlockStorage implementation
    override val encoding: TensorEncoding get() = TensorEncoding.Q5_K
    override val blockSize: Int get() = Q5_KTensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        for (subBlockIdx in 0 until Q5_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val offset = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q5_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q5_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                val outIdx = outputOffset + elementIdx
                if (outIdx >= output.size) return
                val globalIdx = blockIdx * Q5_KTensorData.BLOCK_SIZE + elementIdx
                if (globalIdx >= shape.volume) return
                val code = getCode(blockIdx, elementIdx)
                output[outIdx] = code * scale - offset
            }
        }
    }

    init {
        val requiredBytes = blockCount * Q5_KTensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockD(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val offset = blockIdx * Q5_KTensorData.BYTES_PER_BLOCK
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getBlockDMin(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        val offset = blockIdx * Q5_KTensorData.BYTES_PER_BLOCK + 2
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q5_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        return getBlockD(blockIdx) * getScaleIndex(blockIdx, subBlockIdx)
    }

    override fun getSubBlockMin(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q5_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        return getBlockDMin(blockIdx) * getMinIndex(blockIdx, subBlockIdx)
    }

    /**
     * Port of `get_scale_min_k4` from ggml-quants.c — identical to Q4_K. The
     * 12 scale bytes don't pack 12 bits sequentially per sub-block; sub-blocks
     * 4..7 reuse the top 2 bits of bytes for sub-blocks 0..3.
     */
    private fun getScaleIndex(blockIdx: Int, subBlockIdx: Int): Int {
        val base = blockIdx * Q5_KTensorData.BYTES_PER_BLOCK + 4
        val j = subBlockIdx
        return if (j < 4) {
            data[base + j].toInt() and 0x3F
        } else {
            val low4 = data[base + j + 4].toInt() and 0x0F
            val high2 = (data[base + j - 4].toInt() and 0xFF) ushr 6
            low4 or (high2 shl 4)
        }
    }

    private fun getMinIndex(blockIdx: Int, subBlockIdx: Int): Int {
        val base = blockIdx * Q5_KTensorData.BYTES_PER_BLOCK + 4
        val j = subBlockIdx
        return if (j < 4) {
            data[base + j + 4].toInt() and 0x3F
        } else {
            val low4 = (data[base + j + 4].toInt() and 0xFF) ushr 4
            val high2 = (data[base + j].toInt() and 0xFF) ushr 6
            low4 or (high2 shl 4)
        }
    }

    /**
     * Look up the 5-bit code for `elementIdx` (0..255) within block `blockIdx`.
     * The low 4 bits come from the strided `qs` nibble layout (identical to
     * Q4_K); the 5th bit comes from `qh[l]` where `l` is the intra-group
     * position and the bit index is `2*group` (lo nibble) or `2*group + 1`
     * (hi nibble). Matches `DequantOps.dequantQ5KFromBytes`.
     */
    override fun getCode(blockIdx: Int, elementIdx: Int): Int {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(elementIdx in 0 until Q5_KTensorData.BLOCK_SIZE) {
            "Element index $elementIdx out of bounds (0..255)"
        }
        val base = blockIdx * Q5_KTensorData.BYTES_PER_BLOCK
        val groupIdx = elementIdx / 64           // 0..3 — the `outer` iteration
        val withinGroup = elementIdx % 64        // 0..63
        val l = withinGroup % 32                 // 0..31 — intra-group position
        val qsByte = data[base + Q5_KTensorData.QS_OFFSET + groupIdx * 32 + l].toInt() and 0xFF
        val low = if (withinGroup < 32) qsByte and 0x0F else qsByte ushr 4
        val qhByte = data[base + Q5_KTensorData.QH_OFFSET + l].toInt() and 0xFF
        val bit = if (withinGroup < 32) 2 * groupIdx else 2 * groupIdx + 1
        val fifth = (qhByte ushr bit) and 0x01
        return low or (fifth shl 4)
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q5_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q5_KTensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx).toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        throw UnsupportedOperationException("Q5_K packed tensor data is read-only")
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
        /**
         * Create Q5_KTensorData from raw GGUF bytes.
         */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q5_KBlockTensorData {
            return Q5_KBlockTensorData(shape, bytes)
        }

        /**
         * Convert f16 bits to float32.
         */
        internal fun halfToFloat(hbits: Int): Float {
            val sign = (hbits and 0x8000) shl 16
            val exp = (hbits and 0x7C00) shr 10
            val mant = hbits and 0x03FF

            return when (exp) {
                0 -> {
                    if (mant == 0) {
                        Float.fromBits(sign)
                    } else {
                        var m = mant
                        var e = -14
                        while ((m and 0x400) == 0) {
                            m = m shl 1
                            e--
                        }
                        m = m and 0x3FF
                        val floatExp = (e + 127) shl 23
                        val floatMant = m shl 13
                        Float.fromBits(sign or floatExp or floatMant)
                    }
                }
                31 -> {
                    val floatExp = 0xFF shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
                else -> {
                    val floatExp = (exp - 15 + 127) shl 23
                    val floatMant = mant shl 13
                    Float.fromBits(sign or floatExp or floatMant)
                }
            }
        }
    }
}

/**
 * Dequantize Q5_K tensor data to a FloatArray (canonical ggml formula:
 * `output[i] = code[i] * scale - offset`).
 */
public fun Q5_KTensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    var outIdx = 0
    for (blockIdx in 0 until blockCount) {
        for (subBlockIdx in 0 until Q5_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val offset = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q5_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q5_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                if (outIdx >= shape.volume) break
                val code = getCode(blockIdx, elementIdx)
                result[outIdx++] = code * scale - offset
            }
        }
    }
    return result
}
