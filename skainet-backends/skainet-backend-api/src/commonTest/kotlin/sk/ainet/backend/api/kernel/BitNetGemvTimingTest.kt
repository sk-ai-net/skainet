package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.TimeSource

/**
 * The **fallback**'s cost, measured where the test runs (#1041, M2-A2).
 *
 * `bitnet_gemv/reference` is what a device gets when the NEON artifact is absent, so its time is
 * the denominator of the "NEON is N× faster" claim. This prints milliseconds per call for a fixed
 * shape and asserts only that it produced numbers — a speed assertion here would be a flake on
 * shared CI hardware, and the acceptance measurement belongs on the reference device, where this
 * same test is run from the Kotlin/Native binary.
 */
class BitNetGemvTimingTest {

    private companion object {
        const val K = 1024      // four TQ2_0 blocks per row
        const val N = 256
        const val CALLS = 5
    }

    @Test
    fun referenceKernelThroughput() {
        var seed = 7
        val weightValues = FloatArray(N * K) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val bytes = TernaryCodec.encode(TensorEncoding.TQ2_0, weightValues)
        val weight = TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(N, K), TensorEncoding.TQ2_0,
            TernaryBlockDecoder(TensorEncoding.TQ2_0),
        )
        val activationFloats = FloatArray(K) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 2000 - 1000) / 1000f
        }
        val activation = I8Absmax.requantize(
            TensorView.dense(Storage.Heap.wrap(activationFloats), Shape(1, K), FP32),
            Scope.Ambient,
        )
        val out = TensorView.dense(Storage.Heap.floats(N), Shape(1, N), FP32)
        val kernel = BitNetGemvKernel(BitNetGemvKernel.keyFor(weight.format))

        kernel.run(listOf(activation, weight), out)          // warm up
        val mark = TimeSource.Monotonic.markNow()
        repeat(CALLS) { kernel.run(listOf(activation, weight), out) }
        val perCall = mark.elapsedNow().inWholeMicroseconds / CALLS.toDouble() / 1000.0

        println("[bitnet_gemv] reference k=$K n=$N: $perCall ms/call (${N.toLong() * K} MACs)")
        assertTrue(perCall >= 0.0)
        var sawNonZero = false
        for (o in 0 until N) if (out.get(0, o) != 0f) { sawNonZero = true; break }
        assertTrue(sawNonZero, "the timed kernel must actually compute something")
    }
}
