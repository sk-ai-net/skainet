package sk.ainet.exec.golden

import sk.ainet.exec.golden.GoldenSupport.Packed
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test

/**
 * SKEEP-003 golden gate, decode half: the block decoders of every packed `TensorData` must produce
 * bit-identical floats for the same bytes. Guards the TensorData → TensorView façade migration
 * (M1) and every later refactor of the packed storage types.
 */
class PackedDecodeGoldenTest {

    private companion object {
        const val ROWS = 4
        const val BLOCKS_PER_ROW = 3
        const val SEED = 0x5EED_0001L
    }

    private fun decodeAll(storage: PackedBlockStorage, elementCount: Int, blockSize: Int): FloatArray {
        val out = FloatArray(elementCount)
        val blocks = (elementCount + blockSize - 1) / blockSize
        val tmp = FloatArray(blockSize)
        for (b in 0 until blocks) {
            storage.dequantizeBlock(b, tmp, 0)
            val n = minOf(blockSize, elementCount - b * blockSize)
            tmp.copyInto(out, b * blockSize, 0, n)
        }
        return out
    }

    private fun golden(p: Packed, build: (Shape, ByteArray) -> PackedBlockStorage) {
        val blocks = GoldenSupport.weightBlocks(p, ROWS, BLOCKS_PER_ROW, SEED)
        val shape = Shape(ROWS, BLOCKS_PER_ROW * p.blockSize)
        val storage = build(shape, GoldenSupport.rowMajor(blocks))
        val decoded = decodeAll(storage, shape.volume, p.blockSize)
        GoldenSupport.check("decode/${p.name}", GoldenSupport.digest(decoded))
    }

    @Test fun q4_0() = golden(Packed.Q4_0) { s, b -> Q4_0BlockTensorData(s, b) }
    @Test fun q5_0() = golden(Packed.Q5_0) { s, b -> Q5_0BlockTensorData(s, b) }
    @Test fun q5_1() = golden(Packed.Q5_1) { s, b -> Q5_1BlockTensorData(s, b) }
    @Test fun q8_0() = golden(Packed.Q8_0) { s, b -> Q8_0BlockTensorData(s, b) }
    @Test fun q4_K() = golden(Packed.Q4_K) { s, b -> Q4_KBlockTensorData(s, b) }
    @Test fun q5_K() = golden(Packed.Q5_K) { s, b -> Q5_KBlockTensorData(s, b) }
    @Test fun q6_K() = golden(Packed.Q6_K) { s, b -> Q6_KBlockTensorData(s, b) }

    @Test
    fun ternaryFromValues() {
        val rng = GoldenSupport.Rng(SEED + 7)
        val n = 4 * 96
        val values = ByteArray(n) { (rng.nextByte().toInt() % 3).toByte() } // -2..2 → clamp to {-1,0,1}
        for (i in values.indices) values[i] = values[i].toInt().coerceIn(-1, 1).toByte()
        val t = Ternary2BitTensorData.fromTernaryValues(Shape(4, 96), values, scale = 0.8125f)
        val decoded = decodeAll(t, n, t.blockSize)
        GoldenSupport.check("decode/TERNARY_values", GoldenSupport.digest(decoded))
        GoldenSupport.check("packed/TERNARY_values", GoldenSupport.digest(t.packedData))
    }

    /**
     * #1033: the ternary encodings decode through [TernaryCodec], whose layout is the GGML one
     * (`dequantize_row_tq{1,2}_0`, interleave included). Golden over *encoded* ternary values, so a
     * change to either half of the codec — the packer or the unpacker — moves the digest.
     */
    private fun ternaryGolden(encoding: TensorEncoding, name: String, elements: Int) {
        val rng = GoldenSupport.Rng(SEED + 23)
        val values = FloatArray(elements) { (((rng.nextLong() ushr 40).toInt() and 0xFFFF) % 3 - 1) * 0.5f }
        val bytes = TernaryCodec.encode(encoding, values)
        GoldenSupport.check("packed/$name", GoldenSupport.digest(bytes))
        GoldenSupport.check("decode/$name", GoldenSupport.digest(TernaryCodec.decode(encoding, bytes, elements)))
    }

    @Test fun tq1_0() = ternaryGolden(TensorEncoding.TQ1_0, "TQ1_0", 256 * 3)
    @Test fun tq2_0() = ternaryGolden(TensorEncoding.TQ2_0, "TQ2_0", 256 * 3)
    @Test fun bitnet() = ternaryGolden(TensorEncoding.BITNET_B1_58, "BITNET_B1_58", 256 * 3)

    /** A ternary view decodes exactly like the codec it is driven by — the M2 kernel entry path. */
    @Test
    fun ternaryThroughATensorView() {
        val rng = GoldenSupport.Rng(SEED + 29)
        val values = FloatArray(512) { (((rng.nextLong() ushr 40).toInt() and 0xFFFF) % 3 - 1) * 0.5f }
        for (encoding in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0)) {
            val bytes = TernaryCodec.encode(encoding, values)
            val view = TensorView.packed(Storage.Heap.wrap(bytes), Shape(2, 256), encoding, TernaryBlockDecoder(encoding))
            GoldenSupport.check("view/${encoding.name}", GoldenSupport.digest(view.toFloatArray()))
        }
    }

    @Test
    fun ternaryFromTQ2_0Block() {
        val rng = GoldenSupport.Rng(SEED + 11)
        val block = ByteArray(66) { rng.nextByte() }
        GoldenSupport.le16(block, 64, GoldenSupport.half(0.03125f))
        val t = Ternary2BitTensorData.fromTQ2_0Block(block, Shape(256))
        val decoded = decodeAll(t, 256, t.blockSize)
        GoldenSupport.check("decode/TERNARY_tq2_0", GoldenSupport.digest(decoded))
    }
}
