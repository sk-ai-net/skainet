package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for Q4_K quantized format (canonical ggml layout).
 *
 * Q4_K block format (256 elements per block, 144 bytes per block):
 * - 2 bytes: f16 d (super-block scale)
 * - 2 bytes: f16 dMin (super-block min-scale)
 * - 12 bytes: packed 6-bit scaleIdx + 6-bit minIdx for each of 8 sub-blocks,
 *             encoded with ggml's `get_scale_min_k4` bit-mixing layout (see
 *             ggml-quants.c). Sub-blocks 0..3 take their 6-bit scaleIdx and
 *             minIdx from `scales[j]` and `scales[j+4]`; sub-blocks 4..7
 *             reuse the top 2 bits of earlier scale bytes — *not* a flat
 *             "12 bits per sub-block" packing.
 * - 128 bytes: 4-bit quantized codes, laid out *strided* in 4 groups of 32
 *              bytes. In each 32-byte group the lo nibbles decode to the
 *              first 32 elements of the group's first sub-block, and the hi
 *              nibbles of the *same* bytes decode to the 32 elements of the
 *              group's second sub-block. So byte (j*32 + i) carries
 *              element (2j*32 + i) in its lo nibble and element ((2j+1)*32 + i)
 *              in its hi nibble.
 *
 * Each sub-block s (s=0..7):
 * - 6-bit scaleIdx, 6-bit minIdx (from `get_scale_min_k4`)
 * - scale  = d    * scaleIdx     (no /63 — ggml's `d1 = d * sc`)
 * - offset = dMin * minIdx
 *
 * Dequantization: output[i] = code[i] * scale - offset
 *
 * (Earlier versions of this file used an interleaved `byte[i]→2i,2i+1`
 * codes layout, a flat 12-bits-per-sub-block scale packing, a /63
 * normalisation, and a `+ min` sign — none of which match real GGUF
 * Q4_K_M files. Fixed against `DequantOps.dequantQ4KFromBytes` and
 * the proof in `Q4KCanonicalLayoutTest`.)
 */
public interface Q4_KTensorData : TensorData<DType, Byte> {
    /** Number of Q4_K blocks in the tensor. */
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

    /** Get a 4-bit quantized code value (0..255 elements within block). */
    public fun getCode(blockIdx: Int, elementIdx: Int): Int

    public companion object {
        /** Elements per Q4_K block. */
        public const val BLOCK_SIZE: Int = 256

        /** Elements per sub-block. */
        public const val SUB_BLOCK_SIZE: Int = 32

        /** Number of sub-blocks per block. */
        public const val SUB_BLOCKS_PER_BLOCK: Int = 8

        /** Bytes per Q4_K block (2 + 2 + 12 + 128 = 144). */
        public const val BYTES_PER_BLOCK: Int = 144
    }
}

/**
 * Implementation of Q4_KTensorData backed by a packed byte array (canonical
 * ggml layout — see [Q4_KTensorData] kdoc for the full byte map).
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param packedData the raw packed block data
 */
public class Q4_KBlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q4_KTensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int = (shape.volume + Q4_KTensorData.BLOCK_SIZE - 1) / Q4_KTensorData.BLOCK_SIZE

    // PackedBlockStorage implementation
    override val encoding: TensorEncoding get() = TensorEncoding.Q4_K
    override val blockSize: Int get() = Q4_KTensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        for (subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val offset = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q4_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q4_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                val outIdx = outputOffset + elementIdx
                if (outIdx >= output.size) return
                val globalIdx = blockIdx * Q4_KTensorData.BLOCK_SIZE + elementIdx
                if (globalIdx >= shape.volume) return
                val code = getCode(blockIdx, elementIdx)
                output[outIdx] = code * scale - offset
            }
        }
    }

    init {
        val requiredBytes = blockCount * Q4_KTensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockD(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds (0..$blockCount)" }
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getBlockDMin(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        val offset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 2
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return halfToFloat(halfBits)
    }

    override fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        return getBlockD(blockIdx) * getScaleIndex(blockIdx, subBlockIdx)
    }

    override fun getSubBlockMin(blockIdx: Int, subBlockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            "Sub-block index $subBlockIdx out of bounds (0..7)"
        }
        return getBlockDMin(blockIdx) * getMinIndex(blockIdx, subBlockIdx)
    }

    /**
     * Port of `get_scale_min_k4` from ggml-quants.c. The 12 scale bytes don't
     * pack 12 bits sequentially per sub-block — sub-blocks 4..7 reuse the top
     * 2 bits of bytes for sub-blocks 0..3.
     */
    private fun getScaleIndex(blockIdx: Int, subBlockIdx: Int): Int {
        val base = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 4
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
        val base = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 4
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
     * Look up the 4-bit code for `elementIdx` (0..255) within block
     * `blockIdx`, using ggml's strided per-32-byte-group layout: each
     * 32-byte qs group covers 64 elements, with byte `i` of the group
     * holding element `groupBase + i` in its lo nibble and element
     * `groupBase + i + 32` in its hi nibble.
     */
    override fun getCode(blockIdx: Int, elementIdx: Int): Int {
        require(blockIdx in 0 until blockCount) { "Block index $blockIdx out of bounds" }
        require(elementIdx in 0 until Q4_KTensorData.BLOCK_SIZE) {
            "Element index $elementIdx out of bounds (0..255)"
        }
        val groupIdx = elementIdx / 64           // 0..3 — which 32-byte qs group
        val withinGroup = elementIdx % 64        // 0..63
        val byteOffset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 16 +
            groupIdx * 32 + (withinGroup % 32)
        val codeByte = data[byteOffset].toInt() and 0xFF
        return if (withinGroup < 32) codeByte and 0x0F else codeByte ushr 4
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_KTensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx).toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q4_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q4_KTensorData.BLOCK_SIZE
        val groupIdx = elementIdx / 64
        val withinGroup = elementIdx % 64
        val byteOffset = blockIdx * Q4_KTensorData.BYTES_PER_BLOCK + 16 +
            groupIdx * 32 + (withinGroup % 32)
        val currentByte = data[byteOffset].toInt() and 0xFF
        val newValue = value.toInt() and 0x0F
        data[byteOffset] = if (withinGroup < 32) {
            ((currentByte and 0xF0) or newValue).toByte()
        } else {
            ((currentByte and 0x0F) or (newValue shl 4)).toByte()
        }
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
         * Create Q4_KTensorData from raw GGUF bytes.
         */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q4_KBlockTensorData {
            return Q4_KBlockTensorData(shape, bytes)
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
 * Dequantize Q4_K tensor data to a FloatArray (canonical ggml formula:
 * `output[i] = code[i] * scale - offset`).
 */
public fun Q4_KTensorData.toFloatArray(): FloatArray {
    val result = FloatArray(shape.volume)
    var outIdx = 0
    for (blockIdx in 0 until blockCount) {
        for (subBlockIdx in 0 until Q4_KTensorData.SUB_BLOCKS_PER_BLOCK) {
            val scale = getSubBlockScale(blockIdx, subBlockIdx)
            val offset = getSubBlockMin(blockIdx, subBlockIdx)
            val elemsStart = subBlockIdx * Q4_KTensorData.SUB_BLOCK_SIZE
            for (j in 0 until Q4_KTensorData.SUB_BLOCK_SIZE) {
                val elementIdx = elemsStart + j
                if (outIdx >= shape.volume) break
                val code = getCode(blockIdx, elementIdx)
                result[outIdx++] = code * scale - offset
            }
        }
    }
    return result
}
