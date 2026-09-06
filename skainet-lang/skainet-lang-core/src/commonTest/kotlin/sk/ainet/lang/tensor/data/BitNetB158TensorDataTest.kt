package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * #1140: the first packed ternary `TensorData`. Its decode must be pinned to [TernaryCodec] —
 * the same reference the kernels and the GGUF loader are defined against — so all three readers
 * agree about the same bytes.
 */
class BitNetB158TensorDataTest {

    private fun ternaryValues(count: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(count) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1) * 0.5f
        }
    }

    @Test
    fun roundTripsThroughTheCodec() {
        val n = 6; val k = 32
        val values = ternaryValues(n * k, seed = 5)
        val data = BitNetB158TensorData.fromFloats(Shape(n, k), values)

        assertEquals(TernaryCodec.bitNetScale(data.packedData, n * k), data.scale)
        assertContentEquals(
            TernaryCodec.decodeBitNet(data.packedData, n * k),
            data.toFloatArray(),
            "toFloatArray must equal the codec's decode of the same bytes",
        )
    }

    @Test
    fun getReturnsSignedCodesAndSetWritesThem() {
        val data = BitNetB158TensorData.fromFloats(Shape(8), floatArrayOf(1f, -1f, 0f, 1f, -1f, 0f, 0f, 1f))
        assertEquals(1, data.get(0).toInt())
        assertEquals(-1, data.get(1).toInt())
        assertEquals(0, data.get(2).toInt())
        data.set(2, value = -1)
        assertEquals(-1, data.get(2).toInt())
        assertFailsWith<IllegalArgumentException> { data.set(0, value = 2) }
    }

    @Test
    fun viewCarriesTheExactDispatchFormat() {
        val data = BitNetB158TensorData.fromFloats(Shape(4, 8), ternaryValues(32, seed = 9))
        val view = data.packedView
        assertEquals(TensorEncoding.BITNET_B1_58, view.format.encoding)
        assertEquals(1, data.blockCount, "per-tensor encoding: one block")
        assertEquals(32, data.blockSize)
        // the view decodes with the scale applied, exactly like the codec
        val decoded = TernaryCodec.decodeBitNet(data.packedData, 32)
        assertEquals(decoded[9], view.get(1, 1))
    }

    @Test
    fun rejectsBuffersWithoutTheScaleTrailer() {
        assertFailsWith<IllegalArgumentException> {
            BitNetB158TensorData(Shape(8), ByteArray(2)) // payload only, no trailer
        }
    }
}
