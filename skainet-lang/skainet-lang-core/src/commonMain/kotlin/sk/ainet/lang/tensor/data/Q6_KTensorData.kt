package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Tensor data interface for Q6_K quantized format (ggml/llama.cpp).
 *
 * Q6_K block format (256 elements per block, 210 bytes per block):
 *   - 128 bytes: ql — low 4 bits of each 6-bit code (packed 2-per-byte)
 *   -  64 bytes: qh — high 2 bits of each 6-bit code (packed 4-per-byte)
 *   -  16 bytes: scales — one signed int8 per 16-element sub-block
 *   -   2 bytes: f16 d — block-level scale
 *
 * 16 sub-blocks of 16 elements each. Dequant math per element:
 *   code_4 = low nibble from ql
 *   code_2 = 2-bit slice from qh
 *   code_signed = (code_4 | (code_2 shl 4)) - 32     // 6-bit signed
 *   output = d * scales[sub_block] * code_signed
 *
 * Byte order inside a block matches ggml's struct block_q6_K. The exact
 * packing layout (which ql/qh byte holds which element) is
 * half-interleaved; see [Q6_KBlockTensorData.toFloatArray] or
 * [DequantOps.dequantQ6KFromBytes] for the unrolled reference.
 */
public interface Q6_KTensorData : TensorData<DType, Byte> {
    /** Number of Q6_K blocks in the tensor. */
    public val blockCount: Int

    /** Raw packed data containing all blocks. */
    public val packedData: ByteArray

    /** Get the main scale factor (d) for a block as FP32. */
    public fun getBlockD(blockIdx: Int): Float

    /** Get the signed int8 scale for a specific sub-block within a block. */
    public fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Int

    /** Get the signed 6-bit code for an element. */
    public fun getCode(blockIdx: Int, elementIdx: Int): Int

    public companion object {
        /** Elements per Q6_K block. */
        public const val BLOCK_SIZE: Int = 256

        /** Elements per sub-block. */
        public const val SUB_BLOCK_SIZE: Int = 16

        /** Number of sub-blocks per block. */
        public const val SUB_BLOCKS_PER_BLOCK: Int = 16

        /** Bytes per Q6_K block (128 ql + 64 qh + 16 scales + 2 d = 210). */
        public const val BYTES_PER_BLOCK: Int = 210

        /** Offset (in bytes) of the ql region inside a block. */
        internal const val OFFSET_QL: Int = 0

        /** Offset (in bytes) of the qh region inside a block. */
        internal const val OFFSET_QH: Int = 128

        /** Offset (in bytes) of the scales region inside a block. */
        internal const val OFFSET_SCALES: Int = 192

        /** Offset (in bytes) of the f16 d value inside a block. */
        internal const val OFFSET_D: Int = 208
    }
}

/**
 * Implementation of [Q6_KTensorData] backed by a packed byte array.
 *
 * Memory layout per block (210 bytes) matches ggml's block_q6_K:
 *   - bytes [  0..127]: ql
 *   - bytes [128..191]: qh
 *   - bytes [192..207]: scales (signed int8 each)
 *   - bytes [208..209]: f16 d (little-endian)
 *
 * @param initialShape the logical shape of the tensor (in elements, not blocks)
 * @param packedData the raw packed block data
 */
public class Q6_KBlockTensorData(
    initialShape: Shape,
    private val data: ByteArray,
    /**
     * Which order [data]'s blocks are physically in (#1120/#1124). `ROW_MAJOR` — canonical, as
     * every GGUF-shaped producer writes — unless this weight was relayouted for the packed kernels.
     */
    override val blockOrder: sk.ainet.lang.memory.BlockOrder = sk.ainet.lang.memory.BlockOrder.ROW_MAJOR,
) : Q6_KTensorData, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()
    override val packedData: ByteArray get() = data

    override val blockCount: Int =
        (shape.volume + Q6_KTensorData.BLOCK_SIZE - 1) / Q6_KTensorData.BLOCK_SIZE

    // PackedBlockStorage integration (mirrors Q4_KBlockTensorData).
    override val encoding: TensorEncoding get() = TensorEncoding.Q6_K
    override val blockSize: Int get() = Q6_KTensorData.BLOCK_SIZE

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until blockCount) {
            "Block index $blockIdx out of bounds (0..$blockCount)"
        }
        val base = blockIdx * Q6_KTensorData.BYTES_PER_BLOCK
        val d = getBlockD(blockIdx)
        val qlBase0 = base + Q6_KTensorData.OFFSET_QL
        val qhBase0 = base + Q6_KTensorData.OFFSET_QH
        val scBase0 = base + Q6_KTensorData.OFFSET_SCALES

        // Same half-interleaved unpack the ggml reference uses. See
        // DequantOps.dequantQ6KFromBytes for the golden reference.
        for (half in 0..1) {
            val qlBase = qlBase0 + half * 64
            val qhBase = qhBase0 + half * 32
            val scBase = scBase0 + half * 8
            val outBase = outputOffset + half * 128
            for (l in 0 until 32) {
                val isIdx = l / 16

                val ql0 = data[qlBase + l].toInt() and 0xFF
                val ql32 = data[qlBase + l + 32].toInt() and 0xFF
                val qhL = data[qhBase + l].toInt() and 0xFF

                val q1Low = ql0 and 0x0F
                val q1High = qhL and 0x03
                val q1 = (q1Low or (q1High shl 4)) - 32

                val q2Low = ql32 and 0x0F
                val q2High = (qhL ushr 2) and 0x03
                val q2 = (q2Low or (q2High shl 4)) - 32

                val q3Low = ql0 ushr 4
                val q3High = (qhL ushr 4) and 0x03
                val q3 = (q3Low or (q3High shl 4)) - 32

                val q4Low = ql32 ushr 4
                val q4High = (qhL ushr 6) and 0x03
                val q4 = (q4Low or (q4High shl 4)) - 32

                val sc1 = data[scBase + isIdx + 0].toInt()  // signed int8
                val sc2 = data[scBase + isIdx + 2].toInt()
                val sc3 = data[scBase + isIdx + 4].toInt()
                val sc4 = data[scBase + isIdx + 6].toInt()

                val idxBase = outBase + l
                val vol = shape.volume
                val blockElemBase = blockIdx * Q6_KTensorData.BLOCK_SIZE
                if (blockElemBase + half * 128 + l + 0 < vol)  output[idxBase +  0] = d * sc1 * q1
                if (blockElemBase + half * 128 + l + 32 < vol) output[idxBase + 32] = d * sc2 * q2
                if (blockElemBase + half * 128 + l + 64 < vol) output[idxBase + 64] = d * sc3 * q3
                if (blockElemBase + half * 128 + l + 96 < vol) output[idxBase + 96] = d * sc4 * q4
            }
        }
    }

    init {
        val requiredBytes = blockCount * Q6_KTensorData.BYTES_PER_BLOCK
        require(data.size >= requiredBytes) {
            "Data size ${data.size} is less than required $requiredBytes bytes for $blockCount blocks"
        }
    }

    override fun getBlockD(blockIdx: Int): Float {
        require(blockIdx in 0 until blockCount) {
            "Block index $blockIdx out of bounds (0..$blockCount)"
        }
        val offset = blockIdx * Q6_KTensorData.BYTES_PER_BLOCK + Q6_KTensorData.OFFSET_D
        val b0 = data[offset].toInt() and 0xFF
        val b1 = data[offset + 1].toInt() and 0xFF
        val halfBits = (b1 shl 8) or b0
        return Q4_KBlockTensorData.halfToFloat(halfBits)
    }

    override fun getSubBlockScale(blockIdx: Int, subBlockIdx: Int): Int {
        require(blockIdx in 0 until blockCount)
        require(subBlockIdx in 0 until Q6_KTensorData.SUB_BLOCKS_PER_BLOCK)
        val offset = blockIdx * Q6_KTensorData.BYTES_PER_BLOCK + Q6_KTensorData.OFFSET_SCALES + subBlockIdx
        return data[offset].toInt()  // signed byte → int, preserves sign
    }

    override fun getCode(blockIdx: Int, elementIdx: Int): Int {
        require(blockIdx in 0 until blockCount)
        require(elementIdx in 0 until Q6_KTensorData.BLOCK_SIZE)
        // Same unpack as dequantizeBlock, but just one element.
        val base = blockIdx * Q6_KTensorData.BYTES_PER_BLOCK
        val half = elementIdx / 128
        val within = elementIdx % 128      // 0..127
        val slot = within / 32             // 0..3, which q1..q4
        val l = within % 32                // 0..31
        val qlBase = base + Q6_KTensorData.OFFSET_QL + half * 64
        val qhBase = base + Q6_KTensorData.OFFSET_QH + half * 32
        val qhByte = data[qhBase + l].toInt() and 0xFF
        return when (slot) {
            0 -> {
                val ql0 = data[qlBase + l].toInt() and 0xFF
                val high = qhByte and 0x03
                ((ql0 and 0x0F) or (high shl 4)) - 32
            }
            1 -> {
                val ql32 = data[qlBase + l + 32].toInt() and 0xFF
                val high = (qhByte ushr 2) and 0x03
                ((ql32 and 0x0F) or (high shl 4)) - 32
            }
            2 -> {
                val ql0 = data[qlBase + l].toInt() and 0xFF
                val high = (qhByte ushr 4) and 0x03
                ((ql0 ushr 4) or (high shl 4)) - 32
            }
            else -> {
                val ql32 = data[qlBase + l + 32].toInt() and 0xFF
                val high = (qhByte ushr 6) and 0x03
                ((ql32 ushr 4) or (high shl 4)) - 32
            }
        }
    }

    override fun get(vararg indices: Int): Byte {
        val flatIndex = calcFlatIndex(indices)
        val blockIdx = flatIndex / Q6_KTensorData.BLOCK_SIZE
        val elementIdx = flatIndex % Q6_KTensorData.BLOCK_SIZE
        return getCode(blockIdx, elementIdx).toByte()
    }

    override fun set(vararg indices: Int, value: Byte) {
        error("Q6_KBlockTensorData is read-only; bit-packed writes are not implemented.")
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
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): Q6_KBlockTensorData =
            Q6_KBlockTensorData(shape, bytes)
    }
}

/** Dequantize a Q6_K tensor to a FloatArray. Matches DequantOps.dequantQ6KFromBytes. */
public fun Q6_KTensorData.toFloatArray(): FloatArray {
    val out = FloatArray(shape.volume)
    val self = this as Q6_KBlockTensorData
    for (blockIdx in 0 until blockCount) {
        self.dequantizeBlock(blockIdx, out, blockIdx * Q6_KTensorData.BLOCK_SIZE)
    }
    return out
}
