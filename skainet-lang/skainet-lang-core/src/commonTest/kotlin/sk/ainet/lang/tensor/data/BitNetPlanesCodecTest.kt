package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1150: the multi-plane trit residual codec (the Kotlin port of NeoGPU's `hs_mlt_lmhead_encode`).
 * Eight planes of ±0.5-threshold round-to-trit with ×3 residual scaling reconstruct the row
 * within `rowScale / (2·3⁷)` — the format's defined truncation bound.
 */
class BitNetPlanesCodecTest {

    @Test
    fun eightPlanesReconstructWithinTheTruncationBound() {
        val rows = 5; val cols = 64
        val rng = Random(7)
        val values = FloatArray(rows * cols) { (rng.nextFloat() - 0.5f) * 2f }
        val bytes = TernaryCodec.encodeBitNetPlanes(values, rows, cols)
        val decoded = TernaryCodec.decodeBitNetPlanes(bytes, rows, cols)
        for (r in 0 until rows) {
            val scale = TernaryCodec.planesRowScale(bytes, rows, cols, r)
            // bound: residual after 8 trits is at most 0.5/3^7 of the normalized value, plus
            // FP16 rounding of the scale itself — allow a small epsilon on top.
            val bound = scale * (0.5f / 2187f) + 1e-4f
            for (c in 0 until cols) {
                val err = abs(values[r * cols + c] - decoded[r * cols + c])
                assertTrue(err <= bound, "[$r,$c]: |${values[r * cols + c]} - ${decoded[r * cols + c]}| = $err > $bound")
            }
        }
    }

    @Test
    fun planeZeroCarriesTheSignStructure() {
        val cols = 16
        val values = FloatArray(cols) { if (it % 3 == 0) 0.9f else if (it % 3 == 1) -0.9f else 0.0f }
        val bytes = TernaryCodec.encodeBitNetPlanes(values, 1, cols)
        for (c in 0 until cols) {
            val code = ((bytes[c / 4].toInt() and 0xFF) shr ((c % 4) * 2)) and 3
            val expected = when (c % 3) { 0 -> 2; 1 -> 0; else -> 1 } // +1, -1, 0 biased
            assertEquals(expected, code, "plane-0 trit of element $c")
        }
    }

    @Test
    fun rowScaleIsTheRowsAbsMaxAsFp16() {
        val values = floatArrayOf(0.1f, -0.75f, 0.5f, 0.25f, 0f, 0f, 0f, 0f)
        val bytes = TernaryCodec.encodeBitNetPlanes(values, 1, 8)
        assertEquals(0.75f, TernaryCodec.planesRowScale(bytes, 1, 8, 0), "0.75 is FP16-exact")
    }

    @Test
    fun tensorDataDecodesThroughTheSameCodec() {
        val rows = 3; val cols = 32
        val rng = Random(11)
        val values = FloatArray(rows * cols) { rng.nextFloat() - 0.5f }
        val data = BitNetPlanesTensorData.fromFloats(Shape(rows, cols), values)
        assertEquals(rows, data.blockCount)
        assertEquals(cols, data.blockSize)
        assertEquals(TensorEncoding.BITNET_PLANES, data.packedView.format.encoding)
        val viaCodec = TernaryCodec.decodeBitNetPlanes(data.packedData, rows, cols)
        for (r in 0 until rows) for (c in 0 until cols) {
            assertEquals(viaCodec[r * cols + c], data.get(r, c), "[$r,$c]")
        }
    }

    @Test
    fun bufferGeometryHelpersAgree() {
        val n = 6; val k = 32
        val enc = TensorEncoding.BITNET_PLANES
        assertEquals(n * k / 4, enc.planeStrideBytes(n, k))
        assertEquals(8 * n * k / 4, enc.rowScalesByteOffset(n, k))
        assertEquals(8 * n * k / 4 + 2 * n, enc.bufferBytes(n, k))
        assertEquals(
            enc.bufferBytes(n, k),
            TernaryCodec.encodeBitNetPlanes(FloatArray(n * k), n, k).size,
        )
    }
}
