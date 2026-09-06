package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1138: the f32 reference against the one thing it must equal — an FP32 matmul over the
 * *decoded* weight ([TernaryCodec.decodeBitNet]). The codes-dot is exact; only summation
 * order may differ.
 */
class TernaryF32GemvKernelTest {

    private fun ternaryValues(count: Int, seed: Int): FloatArray {
        var s = seed
        return FloatArray(count) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1) * 0.7f
        }
    }

    private fun weight(n: Int, k: Int, seed: Int = 5): TensorView {
        val bytes = TernaryCodec.encodeBitNet(ternaryValues(n * k, seed))
        return TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(n, k), TensorEncoding.BITNET_B1_58,
            TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * k),
        )
    }

    private fun activation(rows: Int, k: Int, seed: Int = 9): TensorView {
        var s = seed
        val floats = FloatArray(rows * k) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2000 - 1000) / 1000f
        }
        return TensorView.dense(Storage.Heap.wrap(floats), Shape(rows, k), FP32)
    }

    @Test
    fun referenceEqualsMatmulOverTheDecodedWeight() {
        val rows = 2; val k = 96; val n = 5
        val w = weight(n, k)
        val a = activation(rows, k)
        val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)

        val bytes = (w.storage as Storage.Heap).bytes!!
        val decoded = TernaryCodec.decodeBitNet(bytes, n * k)
        for (r in 0 until rows) {
            for (o in 0 until n) {
                var want = 0f
                for (i in 0 until k) want += a.get(r, i) * decoded[o * k + i]
                val got = out.get(r, o)
                assertTrue(
                    abs(got - want) <= 1e-4f * maxOf(1f, abs(want)),
                    "[$r,$o]: reference=$got decoded-matmul=$want",
                )
            }
        }
    }

    @Test
    fun kIndivisibleByFourStillMatchesTheDecodedMatmul() {
        // BITNET_B1_58 packs the flattened tensor, so k % 4 != 0 crosses byte
        // boundaries between rows — the reference reads linear codes and must
        // not care. (The native view kernel falls back to this path.)
        val rows = 1; val k = 6; val n = 3
        val w = weight(n, k, seed = 11)
        val a = activation(rows, k, seed = 13)
        val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)

        val decoded = TernaryCodec.decodeBitNet((w.storage as Storage.Heap).bytes!!, n * k)
        for (o in 0 until n) {
            var want = 0f
            for (i in 0 until k) want += a.get(0, i) * decoded[o * k + i]
            assertTrue(abs(out.get(0, o) - want) <= 1e-5f, "[$o]: ${out.get(0, o)} vs $want")
        }
    }

    @Test
    fun mismatchedInnerDimensionsAreRejected() {
        val w = weight(n = 2, k = 8)
        val a = activation(rows = 1, k = 12)
        val out = TensorView.dense(Storage.Heap.floats(2), Shape(1, 2), FP32)
        assertFailsWith<IllegalArgumentException> {
            TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)
        }
    }
}
