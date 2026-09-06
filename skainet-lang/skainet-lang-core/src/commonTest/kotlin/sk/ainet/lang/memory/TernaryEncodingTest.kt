package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1033 (M2-F1/F2): the ternary encodings and the descriptor that drives their reference decoder.
 *
 * The layout assertions are transcribed from `ggml-quants.c` — `quantize_row_tq{1,2}_0_ref` and
 * `dequantize_row_tq{1,2}_0`. They are the point of this test: a ternary format that decodes in
 * the wrong element order still produces plausible-looking numbers, so the interleave is pinned
 * explicitly rather than only round-tripped.
 */
class TernaryEncodingTest {

    private fun ternary(n: Int, seed: Int = 1): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1).toFloat()
        }
    }

    // --- the descriptor ------------------------------------------------------------------------

    @Test
    fun everyBlockSpecAgreesWithTheEncodingsOwnByteCount() {
        val blocked = listOf(
            TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0,
            TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
            TensorEncoding.TQ1_0, TensorEncoding.TQ2_0,
            TensorEncoding.TurboQuantPolar(4, 128), TensorEncoding.TurboQuantPolarQjl(4, 1, 128),
        )
        for (e in blocked) {
            val spec = e.blockSpec ?: error("${e.name} must have a block spec")
            assertEquals(
                spec.bytesPerBlock.toLong(), e.physicalBytes(spec.blockSize.toLong()),
                "${e.name}: spec says ${spec.bytesPerBlock} B per ${spec.blockSize} elements",
            )
            assertTrue(spec.bitsPerElement <= spec.amortisedBitsPerElement, "${e.name}: payload bits exceed the on-disk rate")
        }
    }

    @Test
    fun theTernaryFamilyIsDescribedAsTernaryWithInt8Activations() {
        for (e in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0, TensorEncoding.TernaryPacked, TensorEncoding.BITNET_B1_58)) {
            assertTrue(e.isTernary, "${e.name} must be ternary")
            assertEquals(Format(Int8, TensorEncoding.DENSE_I8_ABSMAX), e.blockSpec?.activation, "${e.name}: W1.58A8")
        }
        assertTrue(!TensorEncoding.Q4_K.isTernary)
        assertNull(TensorEncoding.Dense(4).blockSpec, "dense is not block-structured")
        assertNull(TensorEncoding.Opaque("X", 10).blockSpec, "an opaque encoding has no known geometry")
    }

    @Test
    fun theTernaryRatesAreWhatTheNamesClaim() {
        assertEquals(1.625, TensorEncoding.TQ1_0.blockSpec!!.bitsPerElement, "TQ1_0 payload: 52 B per 256 elements")
        assertEquals(1.6875, TensorEncoding.TQ1_0.blockSpec!!.amortisedBitsPerElement, "with the FP16 block scale")
        assertEquals(2.0, TensorEncoding.TQ2_0.blockSpec!!.bitsPerElement)
        assertEquals(2.0625, TensorEncoding.TQ2_0.blockSpec!!.amortisedBitsPerElement)
        assertEquals(ScalePlacement.BLOCK_TAIL, TensorEncoding.TQ2_0.blockSpec!!.scale)
        assertEquals(ScalePlacement.PER_TENSOR, TensorEncoding.BITNET_B1_58.blockSpec!!.scale)
        assertEquals(ScalePlacement.NONE, TensorEncoding.TernaryPacked.blockSpec!!.scale, "its scale lives on the TensorData")
    }

    @Test
    fun thePerTensorEncodingsSizeThemselvesFromTheElementCount() {
        assertEquals(64L + 4, TensorEncoding.BITNET_B1_58.physicalBytes(256))
        assertEquals(64L, TensorEncoding.TernaryPacked.physicalBytes(256))
        assertTrue(TensorEncoding.BITNET_B1_58.blockSpec!!.isPerTensor)
    }

    // --- TQ2_0 ---------------------------------------------------------------------------------

    @Test
    fun tq2_0PacksFourElementsThirtyTwoApartPerByte() {
        // ggml: byte `j + m` of a 32-byte chunk carries element `chunk*128 + l*32 + m` in bit pair `l`.
        val values = FloatArray(256)
        values[0] = -1f; values[32] = 0f; values[64] = 1f; values[96] = -1f
        values[255] = 1f                                    // fixes absmax at 1.0 (exact in FP16)
        val bytes = TernaryCodec.encodeTq2_0(values)
        assertEquals(66, bytes.size)
        val codes = (0 until 4).map { l -> (bytes[0].toInt() shr (2 * l)) and 3 }
        assertEquals(listOf(0, 1, 2, 0), codes, "bit pairs of byte 0 are elements 0, 32, 64, 96")
        assertEquals(1.0f, Fp16Codec.decode((bytes[64].toInt() and 0xFF) or ((bytes[65].toInt() and 0xFF) shl 8)), "the FP16 scale is the block tail")

        // element 128 opens the second 32-byte chunk, not byte 32's second bit pair
        val second = FloatArray(256); second[128] = 1f
        val b2 = TernaryCodec.encodeTq2_0(second)
        assertEquals(2, b2[32].toInt() and 3, "element 128 is byte 32, bit pair 0")
    }

    @Test
    fun tq2_0RoundTripsExactly() {
        val values = ternary(256 * 3, seed = 7).also { it[0] = 1f }
        val decoded = TernaryCodec.decodeTq2_0(TernaryCodec.encodeTq2_0(values), values.size)
        assertTrue(values.contentEquals(decoded), "TQ2_0 must round-trip ternary values exactly")
    }

    // --- TQ1_0 ---------------------------------------------------------------------------------

    @Test
    fun tq1_0PacksFiveBaseThreeDigitsPerByteMostSignificantFirst() {
        // ggml: qs[m] holds elements m, m+32, m+64, m+96, m+128 as base-3 digits, scaled by 256/243.
        val values = FloatArray(256)
        values[0] = 1f                                       // digit 0 (most significant) = 2
        values[128] = -1f                                    // digit 4 = 0
        val bytes = TernaryCodec.encodeTq1_0(values)
        assertEquals(54, bytes.size)
        val v = 2 * 81 + 1 * 27 + 1 * 9 + 1 * 3 + 0          // digits 2,1,1,1,0 → codes for +1,0,0,0,-1
        assertEquals(((v * 256 + 242) / 243).toByte(), bytes[0], "byte 0 is the scaled base-3 word")
        assertEquals(1.0f, Fp16Codec.decode((bytes[52].toInt() and 0xFF) or ((bytes[53].toInt() and 0xFF) shl 8)))

        val decoded = TernaryCodec.decodeTq1_0(bytes, 256)
        assertEquals(1f, decoded[0]); assertEquals(0f, decoded[32]); assertEquals(-1f, decoded[128])
    }

    @Test
    fun tq1_0PutsTheLastSixteenElementsInTheFourQhBytes() {
        val values = FloatArray(256)
        values[240] = 1f; values[244] = -1f; values[255] = 1f
        val bytes = TernaryCodec.encodeTq1_0(values)
        val decoded = TernaryCodec.decodeTq1_0(bytes, 256)
        assertEquals(1f, decoded[240], "qh byte 0, digit 0")
        assertEquals(-1f, decoded[244], "qh byte 0, digit 1")
        assertEquals(1f, decoded[255], "qh byte 3, digit 3 — the last element of the block")
        assertEquals(0f, decoded[241])
    }

    @Test
    fun tq1_0RoundTripsExactly() {
        val values = ternary(256 * 2, seed = 11).also { it[5] = 1f; it[256] = 1f }
        val decoded = TernaryCodec.decodeTq1_0(TernaryCodec.encodeTq1_0(values), values.size)
        assertTrue(values.contentEquals(decoded), "TQ1_0 must round-trip ternary values exactly")
    }

    @Test
    fun tq1_0IsSmallerThanTq2_0WhichIsSmallerThanTwoBitsPlusNothing() {
        val n = 256L * 40
        assertTrue(TensorEncoding.TQ1_0.physicalBytes(n)!! < TensorEncoding.TQ2_0.physicalBytes(n)!!)
        assertEquals(54L * 40, TensorEncoding.TQ1_0.physicalBytes(n))
        assertEquals(66L * 40, TensorEncoding.TQ2_0.physicalBytes(n))
    }

    // --- BitNet b1.58 --------------------------------------------------------------------------

    @Test
    fun bitNetKeepsOneScaleForTheWholeTensorAndFourElementsPerByte() {
        val values = FloatArray(64) { if (it % 3 == 0) 0.75f else if (it % 3 == 1) -0.75f else 0f }
        val bytes = TernaryCodec.encodeBitNet(values)
        assertEquals(64 / 4 + 4, bytes.size)
        val scale = TernaryCodec.bitNetScale(bytes, 64)
        assertTrue(scale > 0f, "absmean scale")
        val decoded = TernaryCodec.decodeBitNet(bytes, 64)
        for (i in values.indices) {
            val expected = when {
                values[i] > 0f -> scale
                values[i] < 0f -> -scale
                else -> 0f
            }
            assertEquals(expected, decoded[i], "element $i")
        }
        // four consecutive elements per byte, low pair first: codes (+1,-1,0,+1) → 2 | 0<<2 | 1<<4 | 2<<6
        assertEquals((2 or (0 shl 2) or (1 shl 4) or (2 shl 6)).toByte(), bytes[0])
    }

    @Test
    fun theSameValuesProduceDifferentBytesInEachTernaryEncoding() {
        val values = ternary(256, seed = 3).also { it[0] = 1f }
        val tq1 = TernaryCodec.encode(TensorEncoding.TQ1_0, values)
        val tq2 = TernaryCodec.encode(TensorEncoding.TQ2_0, values)
        val bit = TernaryCodec.encode(TensorEncoding.BITNET_B1_58, values)
        assertNotEquals(tq1.size, tq2.size)
        assertNotEquals(tq2.size, bit.size)
        for (e in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0)) {
            assertTrue(TernaryCodec.decode(e, TernaryCodec.encode(e, values), 256).contentEquals(values), "${e.name} dispatch round-trip")
        }
        assertFailsWith<IllegalArgumentException> { TernaryCodec.encode(TensorEncoding.TernaryPacked, values) }
        assertFailsWith<IllegalArgumentException> { TernaryCodec.encode(TensorEncoding.Q4_K, values) }
    }

    @Test
    fun encodingRefusesAPartialBlock() {
        assertFailsWith<IllegalArgumentException> { TernaryCodec.encodeTq2_0(FloatArray(100)) }
        assertFailsWith<IllegalArgumentException> { TernaryCodec.encodeTq1_0(FloatArray(300)) }
    }

    // --- through a TensorView ------------------------------------------------------------------

    @Test
    fun aTernaryViewDecodesThroughTheDescriptorDrivenDecoder() {
        val values = ternary(256 * 2, seed = 5).also { it[0] = 1f; it[300] = 1f }
        for (encoding in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0)) {
            val bytes = TernaryCodec.encode(encoding, values)
            val storage = Storage.Heap.wrap(bytes)
            val decoder = TernaryBlockDecoder(encoding)
            val view = TensorView.packed(storage, Shape(2, 256), encoding, decoder)
            assertEquals(Format(FP32, encoding), view.format)
            assertEquals(values[0], view.get(0, 0), "${encoding.name}: first element")
            assertEquals(values[300], view.get(1, 44), "${encoding.name}: second row")
            assertEquals(values[511], view.get(1, 255), "${encoding.name}: last element")
            assertTrue(values.contentEquals(view.toFloatArray()), "${encoding.name}: whole-view decode")
        }
    }

    @Test
    fun perTensorEncodingsAreNotBlockDecoders() {
        assertFailsWith<IllegalArgumentException> { TernaryBlockDecoder(TensorEncoding.TernaryPacked) }
        assertFailsWith<IllegalArgumentException> { TernaryBlockDecoder(TensorEncoding.Q4_K) }
    }
}
