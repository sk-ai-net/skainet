package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The consumer-shaped path for packed ternary weights (#1136): `ops.matmul(x, ops.transpose(W))`
 * — what every dense-layer `linearProject` does — must route a `[out, in]` packed ternary weight
 * through the Wᵀ marker into `KernelDispatch`, never through a dense decode-copy transpose.
 *
 * Before `isHeapPackedWeight` was broadened from the hardcoded Q-format list to
 * `PackedBlockStorage`, `transpose` dense-copied these tensors through their decoding `get()` —
 * which returns ternary CODES, not values — and threw `ClassCastException` (Byte → Float) on the
 * JVM. The oracle here is the decoded matmul.
 */
class TernaryWeightTransposeDispatchTest {

    private val ctx = DirectCpuExecutionContext()

    private fun ternaryValues(count: Int, seed: Int): FloatArray {
        val rng = Random(seed)
        return FloatArray(count) { (rng.nextInt(3) - 1) * 0.25f }
    }

    private fun assertMatmulTransposedMatchesDecode(
        n: Int,
        k: Int,
        weightData: sk.ainet.lang.tensor.data.TensorData<FP32, Float>,
        decoded: FloatArray,
        seed: Int,
        tol: Float,
    ) {
        val w = ctx.fromData(weightData, FP32::class)
        val rng = Random(seed)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(1, k), FP32::class, FloatArray(k) { rng.nextFloat() - 0.5f })

        val out = ctx.ops.matmul(x, ctx.ops.transpose(w))

        for (o in 0 until n) {
            var want = 0f
            for (i in 0 until k) want += x.data.get(0, i) * decoded[o * k + i]
            val got = out.data.get(0, o)
            assertTrue(
                abs(got - want) <= tol * maxOf(1f, abs(want)),
                "[$o]: matmul(x, transpose(W))=$got decoded-matmul=$want",
            )
        }
    }

    @Test
    fun bitNetB158WeightDispatchesThroughTheTransposeMarker() {
        val n = 5; val k = 32
        val values = ternaryValues(n * k, seed = 3)
        val data = BitNetB158TensorData.fromFloats(Shape(n, k), values)
        val decoded = TernaryCodec.decodeBitNet(data.packedData, n * k)
        @Suppress("UNCHECKED_CAST")
        assertMatmulTransposedMatchesDecode(
            n, k, data as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, decoded, seed = 7, tol = 1e-4f,
        )
    }

    @Test
    fun bitNetPlanesWeightDispatchesThroughTheTransposeMarker() {
        val n = 4; val k = 32
        val rng = Random(11)
        val values = FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f }
        val data = BitNetPlanesTensorData.fromFloats(Shape(n, k), values)
        val decoded = TernaryCodec.decodeBitNetPlanes(data.packedData, n, k)
        @Suppress("UNCHECKED_CAST")
        assertMatmulTransposedMatchesDecode(
            n, k, data as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, decoded, seed = 13, tol = 1e-3f,
        )
    }
}
