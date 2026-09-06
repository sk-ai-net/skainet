package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** SKEEP-003 §5.2: platform packs register their kernels under view keys; the reference is always present. */
class KernelPacksTest {

    @AfterTest fun cleanup() { KernelDispatch.clearForTesting(); KernelRegistry.clearForTesting() }

    /** A stand-in pack whose FP32 GEMM is a plain triple loop with the SPI's stride contract. */
    private class FakeProvider(override val name: String = "fake", override val priority: Int = 100) : KernelProvider {
        var calls = 0
        override fun isAvailable(): Boolean = true
        override fun matmulFp32(): Fp32MatmulKernel = object : Fp32MatmulKernel {
            override fun matmul(
                a: FloatArray, aOffset: Int, aStride: Int,
                b: FloatArray, bOffset: Int, bStride: Int,
                out: FloatArray, outOffset: Int, outStride: Int,
                m: Int, n: Int, k: Int,
            ) {
                calls++
                for (i in 0 until m) for (j in 0 until n) {
                    var acc = 0f
                    for (p in 0 until k) acc += a[aOffset + i * aStride + p] * b[bOffset + p * bStride + j]
                    out[outOffset + i * outStride + j] = acc
                }
            }
        }
    }

    private fun view(shape: Shape, values: FloatArray): TensorView =
        TensorView.dense(Storage.Heap.wrap(values), shape, FP32)

    @Test
    fun theReferenceKernelIsAlwaysInstalled() {
        KernelPacks.installReference()
        val dense = OperandKey.contiguous(Format.dense(FP32))
        val k = KernelDispatch.find(KernelKey("matmul", listOf(dense, dense)))
        assertEquals("reference", k?.name)
    }

    @Test
    fun aPackKernelServesTheDenseKeyAndAgreesWithTheReference() {
        val provider = FakeProvider()
        KernelRegistry.register(provider)
        KernelPacks.install(provider)

        val a = view(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        // the weight as SKaiNET stores it, [k, n] = [3, 2], handed to the dispatcher transposed
        val wBuf = floatArrayOf(1f, 0.5f, 2f, 1.5f, 3f, 2.5f)
        val w = view(Shape(3, 2), wBuf).transpose()            // [2, 3] output-major view
        val out = view(Shape(2, 2), FloatArray(4))
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, sink = sink)

        assertTrue(provider.calls > 0, "the pack kernel must have run")
        assertEquals("fake-fp32", assertIs<TraceEvent.KernelRun>(sink.events().single()).kernel)
        // reference numbers
        val expected = FloatArray(4)
        for (i in 0 until 2) for (j in 0 until 2) {
            var acc = 0f
            for (p in 0 until 3) acc += a.get(i, p) * wBuf[p * 2 + j]
            expected[i * 2 + j] = acc
        }
        for (i in expected.indices) assertTrue(abs(out.get(i / 2, i % 2) - expected[i]) < 1e-4f, "element $i: ${out.get(i / 2, i % 2)} vs ${expected[i]}")
    }

    @Test
    fun anOutputMajorWeightFallsBackToTheReferenceInsteadOfBeingMisIndexed() {
        val provider = FakeProvider()
        KernelRegistry.register(provider); KernelPacks.install(provider)
        val a = view(Shape(1, 3), floatArrayOf(1f, 2f, 3f))
        // a genuinely output-major weight [n, k] (not a transposed view): strides [k, 1]
        val w = view(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        val out = view(Shape(1, 2), FloatArray(2))
        KernelDispatch.matmul(a, w, out)
        assertEquals(1f * 1 + 2f * 2 + 3f * 3, out.get(0, 0))   // reference semantics: out = a x wᵀ
        assertEquals(1f * 4 + 2f * 5 + 3f * 6, out.get(0, 1))
    }

    @Test
    fun keysCarryPlatformCapabilities() {
        val dense = OperandKey.contiguous(Format.dense(FP32))
        val plain = KernelKey("matmul", listOf(dense, dense))
        val neon = KernelKey("matmul", listOf(dense, dense), capabilities = setOf("dotprod", KernelPacks.CAPABILITY_VECTOR))
        assertTrue(plain != neon, "capabilities are part of the key")
        assertEquals("matmul(Float32/Dense(4B) contiguous × Float32/Dense(4B) contiguous) @host [dotprod,vector]", neon.toString())
    }
}
