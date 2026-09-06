package sk.ainet.exec.tensor.ops

import sk.ainet.backend.api.kernel.DispatchMode
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * SKEEP-003 §5.1 / PRD M1-F5, M1-A4: the generic matmul path now goes through the kernel registry.
 * The two crash classes this replaces are #993 (a rank-1 decode-step activation reaching a kernel
 * written for rank 2, then `matmulGeneric` reading a packed weight's raw byte) and #991 (an
 * activation whose `TensorData` subtype the fast path did not accept). Both must produce correct,
 * finite numbers — and the registry path and the legacy path must agree.
 */
class RegistryMatmulDispatchTest {

    @AfterTest fun reset() { DispatchMode.overrideEnabled = null }

    private val ctx = DirectCpuExecutionContext()

    private fun half(v: Float): Int {
        val b = v.toRawBits(); val sign = (b ushr 16) and 0x8000
        val e = ((b ushr 23) and 0xFF) - 127 + 15; val m = b and 0x7FFFFF
        if (e <= 0) return sign; if (e >= 31) return sign or 0x7C00
        return sign or (e shl 10) or (m ushr 13)
    }

    /** A Q8_0 weight stored as [k, n] = [32, rows] would be column-major; SKaiNET stores [out, in] and transposes. */
    private fun q8Bytes(rows: Int): ByteArray {
        val bytes = ByteArray(rows * 34)
        for (r in 0 until rows) {
            val off = r * 34; val d = half(0.25f)
            bytes[off] = (d and 0xFF).toByte(); bytes[off + 1] = ((d ushr 8) and 0xFF).toByte()
            for (i in 0 until 32) bytes[off + 2 + i] = ((i - 16) + r * 3).toByte()
        }
        return bytes
    }

    @Suppress("UNCHECKED_CAST")
    private fun packedWeight(rows: Int): Pair<sk.ainet.lang.tensor.Tensor<FP32, Float>, FloatArray> {
        val bytes = q8Bytes(rows)
        val data = Q8_0BlockTensorData(Shape(rows, 32), bytes)
        val t = ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
        return t to data.toFloatArray()
    }

    @Test
    fun rank1DecodeStepAgainstAPackedWeightIsCorrect() {
        // #993: the first post-prefill decode step passes a rank-1 [hidden] activation
        val (w, wf) = packedWeight(4)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(32), FP32::class, FloatArray(32) { (it % 7) * 0.5f })
        val out = ctx.ops.matmulWeightTransposed(x, w)       // [32] x [32, 4] -> [4]
        val got = out.data.copyToFloatArray()
        assertTrue(out.shape.rank == 1 && out.shape[0] == 4, "expected [4], was ${out.shape}")
        for (j in 0 until 4) {
            var expect = 0f
            for (t in 0 until 32) expect += ((t % 7) * 0.5f) * wf[j * 32 + t]
            assertTrue(got[j].isFinite(), "row $j is not finite")
            assertTrue(abs(got[j] - expect) < 1e-2f, "row $j: ${got[j]} vs $expect")
        }
    }

    @Test
    fun registryAndLegacyPathsAgreeOnAPackedWeight() {
        val (w, _) = packedWeight(3)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 32), FP32::class, FloatArray(64) { (it % 5) * 0.25f })
        val wt = ctx.ops.relayoutPackedWeightForKernels(w)

        DispatchMode.overrideEnabled = true
        val viaRegistry = ctx.ops.matmul(x, wt).data.copyToFloatArray()
        DispatchMode.overrideEnabled = false
        val viaLegacy = ctx.ops.matmul(x, wt).data.copyToFloatArray()

        for (i in viaRegistry.indices) {
            assertTrue(viaRegistry[i].isFinite() && viaLegacy[i].isFinite())
            assertTrue(abs(viaRegistry[i] - viaLegacy[i]) < 1e-3f, "element $i: registry ${viaRegistry[i]} vs legacy ${viaLegacy[i]}")
        }
    }

    @Test
    fun batchedActivationsFlattenAndReshape() {
        val (w, wf) = packedWeight(2)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 32), FP32::class, FloatArray(192) { (it % 4).toFloat() })
        val out = ctx.ops.matmulWeightTransposed(x, w)
        assertTrue(out.shape.dimensions.toList() == listOf(2, 3, 2), "expected [2, 3, 2], was ${out.shape}")
        val got = out.data.copyToFloatArray()
        var expect = 0f
        for (t in 0 until 32) expect += ((t % 4).toFloat()) * wf[t]
        assertTrue(abs(got[0] - expect) < 1e-2f, "${got[0]} vs $expect")
    }

    @Test
    fun q4kWeightsDecodeThroughTheRegistry() {
        val bytes = ByteArray(144) { (it * 11).toByte() }
        val d = half(0.01f); bytes[0] = (d and 0xFF).toByte(); bytes[1] = ((d ushr 8) and 0xFF).toByte()
        val dm = half(0.005f); bytes[2] = (dm and 0xFF).toByte(); bytes[3] = ((dm ushr 8) and 0xFF).toByte()
        val data = Q4_KBlockTensorData(Shape(1, 256), bytes)
        @Suppress("UNCHECKED_CAST")
        val w = ctx.fromData(data as TensorData<FP32, Float>, FP32::class)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(256), FP32::class, FloatArray(256) { 0.125f })
        val got = ctx.ops.matmulWeightTransposed(x, w).data.copyToFloatArray()
        var expect = 0f
        for (t in 0 until 256) expect += 0.125f * data.toFloatArray()[t]
        assertTrue(got[0].isFinite()); assertTrue(abs(got[0] - expect) < 1e-2f, "${got[0]} vs $expect")
    }
}
