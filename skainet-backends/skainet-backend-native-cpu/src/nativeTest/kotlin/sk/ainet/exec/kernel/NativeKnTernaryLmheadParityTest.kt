package sk.ainet.exec.kernel

import sk.ainet.lang.memory.TernaryCodec
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Proves the Kotlin/Native cinterop path of the vendored fused lm_head (#1150):
 * [NativeKnTernaryLmhead] against a local reimplementation of the 4-plane fused contract
 * (`out[o] = rowScale[o] · Σ_q (1/3^q) · dot(in, plane_q row o)`). Weights come from the real
 * codec ([TernaryCodec.encodeBitNetPlanes]) so the buffer geometry is the production one.
 *
 * Runs on the host archive AND under qemu-aarch64 (`-PcrossArm64=true`); the C kernel spawns its
 * 4 pthreads at any output_dim, so every case exercises its threading through cinterop.
 */
class NativeKnTernaryLmheadParityTest {

    private fun reference(
        weight: ByteArray, firstPlane: Int, n: Int, k: Int, input: FloatArray,
    ): FloatArray {
        val planeStride = n * k / 4
        val scalesOffset = 8 * planeStride
        val rowBytes = k / 4
        return FloatArray(n) { o ->
            val bits = (weight[scalesOffset + o * 2].toInt() and 0xFF) or
                ((weight[scalesOffset + o * 2 + 1].toInt() and 0xFF) shl 8)
            val scale = sk.ainet.lang.types.Fp16Codec.decode(bits)
            var acc = 0.0
            var w = 1.0
            for (q in 0 until 4) {
                val base = (firstPlane + q) * planeStride + o * rowBytes
                var dot = 0.0
                for (i in 0 until k) {
                    val code = ((weight[base + i / 4].toInt() and 0xFF) shr ((i % 4) * 2)) and 3
                    dot += (code - 1) * input[i]
                }
                acc += dot * w
                w /= 3.0
            }
            (acc * scale).toFloat()
        }
    }

    private fun assertParity(n: Int, k: Int, seed: Int) {
        val rng = Random(seed)
        val values = FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f }
        val weight = TernaryCodec.encodeBitNetPlanes(values, n, k)
        val input = FloatArray(k) { rng.nextFloat() - 0.5f }
        val planeStride = n * k / 4
        val scalesOffset = 8 * planeStride

        for (firstPlane in intArrayOf(0, 4)) {
            val out = FloatArray(n)
            NativeKnTernaryLmhead.lmheadStage1(
                input, 0, weight,
                planesByteOffset = firstPlane * planeStride,
                planeStrideBytes = planeStride,
                rowScaleByteOffset = scalesOffset,
                inputDim = k, outputDim = n,
                out = out, outOffset = 0,
            )
            val expected = reference(weight, firstPlane, n, k, input)
            for (o in 0 until n) {
                val diff = abs(expected[o] - out[o])
                assertTrue(
                    diff <= 1e-3f * maxOf(1f, abs(expected[o])),
                    "planes $firstPlane..${firstPlane + 3} [$o]: reference=${expected[o]} cinterop=${out[o]}",
                )
            }
        }
    }

    @Test fun small_head() = assertParity(n = 8, k = 64, seed = 1)

    @Test fun bitnet_hidden_size() = assertParity(n = 96, k = 2560, seed = 2)

    @Test fun zero_input_dim_zeros_output() {
        val out = FloatArray(3) { 9f }
        NativeKnTernaryLmhead.lmheadStage1(FloatArray(0), 0, ByteArray(8), 0, 0, 0, 0, 3, out, 0)
        for (v in out) assertEquals(0f, v)
    }
}
