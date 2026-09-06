package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.DType

/**
 * Packed `[rows, cols]` weight in the [TensorEncoding.BITNET_PLANES] layout (#1150): 8 trit
 * planes + FP16 per-row scales, decode pinned to [TernaryCodec.decodeBitNetPlanesRow].
 *
 * One block per **row** ([blockSize] = cols): the format's scale granularity is the row, and the
 * fused lm_head kernel consumes whole rows. [packedView] carries
 * `Format(FP32, BITNET_PLANES) × BLOCKED_ROW_MAJOR` — the exact key the planes kernel pack
 * serves. `get` returns the fully decoded value (all 8 planes × row scale); there is no single
 * meaningful "code" per element in a residual format.
 */
public class BitNetPlanesTensorData(
    initialShape: Shape,
    private val data: ByteArray,
) : TensorData<DType, Float>, PackedBlockStorage {

    /** The façade over the packed bytes (SKEEP-003 §4.1): see [PackedBlockStorage.packedView]. */
    override val view: sk.ainet.lang.memory.TensorView get() = packedView

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())

    public val rows: Int
    public val cols: Int

    init {
        require(shape.rank == 2) { "BITNET_PLANES is a [rows, cols] weight format; got rank ${shape.rank}" }
        rows = shape[0]
        cols = shape[1]
        require(cols % 4 == 0) { "BITNET_PLANES needs cols % 4 == 0; got $cols" }
        val required = TensorEncoding.BITNET_PLANES.bufferBytes(rows, cols)
        require(data.size >= required) {
            "BitNetPlanesTensorData: buffer is ${data.size} bytes, need >= $required for [$rows, $cols]"
        }
    }

    override val encoding: TensorEncoding get() = TensorEncoding.BITNET_PLANES
    override val blockCount: Int get() = rows
    override val blockSize: Int get() = cols
    override val packedData: ByteArray get() = data

    /** The FP16 per-row scale of [row]. */
    public fun rowScale(row: Int): Float = TernaryCodec.planesRowScale(data, rows, cols, row)

    override fun dequantizeBlock(blockIdx: Int, output: FloatArray, outputOffset: Int) {
        require(blockIdx in 0 until rows) { "row $blockIdx out of bounds (0..<$rows)" }
        TernaryCodec.decodeBitNetPlanesRow(data, rows, cols, blockIdx, output, outputOffset)
    }

    override fun get(vararg indices: Int): Float {
        require(indices.size == 2) { "BITNET_PLANES data is 2-D" }
        val row = FloatArray(cols)
        TernaryCodec.decodeBitNetPlanesRow(data, rows, cols, indices[0], row, 0)
        return row[indices[1]]
    }

    override fun set(vararg indices: Int, value: Float) {
        throw UnsupportedOperationException(
            "BITNET_PLANES is a residual format — single elements cannot be re-encoded in place; " +
                "re-encode the tensor with TernaryCodec.encodeBitNetPlanes",
        )
    }

    public companion object {
        /** Wrap raw plane bytes (validates the size against the shape). */
        public fun fromRawBytes(shape: Shape, bytes: ByteArray): BitNetPlanesTensorData =
            BitNetPlanesTensorData(shape, bytes)

        /** Encode [values] (row-major `[rows, cols]`) with [TernaryCodec.encodeBitNetPlanes]. */
        public fun fromFloats(shape: Shape, values: FloatArray): BitNetPlanesTensorData {
            require(shape.rank == 2) { "BITNET_PLANES is a [rows, cols] weight format" }
            return BitNetPlanesTensorData(shape, TernaryCodec.encodeBitNetPlanes(values, shape[0], shape[1]))
        }
    }
}
