package sk.ainet.backend.api.kernel

import java.lang.foreign.ValueLayout
import sk.ainet.lang.memory.SegmentStorage
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * A dense-FP32 weight whose storage is [SegmentStorage] (the shape a dequantized, mmap-context
 * GGUF weight actually has — e.g. Gemma 4's `per_layer_model_proj.weight`) must still reach the
 * pack's fast kernel, not fall to the decoding reference. Before this fix `Fp32ViewMatmulKernel`
 * required `Storage.Heap` and fell back unconditionally otherwise — correct, but ~1000x slower
 * (the crash this masked until #325/#341's readDense fix is a separate, already-fixed bug: the
 * reference kernel didn't even support element access over Segment storage at all).
 */
class KernelPacksSegmentStorageTest {

    @AfterTest fun cleanup() { KernelDispatch.clearForTesting(); KernelRegistry.clearForTesting() }

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

    private fun segmentWeight(shape: Shape, values: FloatArray): TensorView {
        val s = SegmentStorage.allocate(bytes = values.size.toLong() * 4)
        for (i in values.indices) s.segment().setAtIndex(ValueLayout.JAVA_FLOAT, i.toLong(), values[i])
        return TensorView.dense(s, shape, FP32)
    }

    private fun heapView(shape: Shape, values: FloatArray): TensorView =
        TensorView.dense(Storage.Heap.wrap(values), shape, FP32)

    @Test
    fun aSegmentBackedWeightStillReachesThePackKernel() {
        val provider = FakeProvider()
        KernelRegistry.register(provider)
        KernelPacks.install(provider)

        val a = heapView(Shape(2, 3), floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f))
        // weight as stored [k, n] = [3, 2], handed to the dispatcher transposed -> [2, 3] view,
        // strides[0] == 1 (the case Fp32ViewMatmulKernel actually serves)
        val wBuf = floatArrayOf(1f, 0.5f, 2f, 1.5f, 3f, 2.5f)
        val w = segmentWeight(Shape(3, 2), wBuf).transpose()
        val out = heapView(Shape(2, 2), FloatArray(4))

        KernelDispatch.matmul(a, w, out)

        assertTrue(provider.calls > 0, "the pack kernel must have run, not the reference fallback")
        val expected = FloatArray(4)
        for (i in 0 until 2) for (j in 0 until 2) {
            var acc = 0f
            for (p in 0 until 3) acc += a.get(i, p) * wBuf[p * 2 + j]
            expected[i * 2 + j] = acc
        }
        for (i in expected.indices) {
            assertTrue(abs(out.get(i / 2, i % 2) - expected[i]) < 1e-4f, "element $i: ${out.get(i / 2, i % 2)} vs ${expected[i]}")
        }
    }

    @Test
    fun aSegmentBackedActivationStillReachesThePackKernel() {
        val provider = FakeProvider()
        KernelRegistry.register(provider)
        KernelPacks.install(provider)

        val a = segmentWeight(Shape(1, 3), floatArrayOf(1f, 2f, 3f))
        val wBuf = floatArrayOf(1f, 0.5f, 2f, 1.5f, 3f, 2.5f)
        val w = heapView(Shape(3, 2), wBuf).transpose()
        val out = heapView(Shape(1, 2), FloatArray(2))

        KernelDispatch.matmul(a, w, out)

        assertTrue(provider.calls > 0, "the pack kernel must have run, not the reference fallback")
        assertTrue(abs(out.get(0, 0) - (1f * 1 + 2f * 2 + 3f * 3)) < 1e-4f)
        assertTrue(abs(out.get(0, 1) - (1f * 0.5f + 2f * 1.5f + 3f * 2.5f)) < 1e-4f)
    }
}
