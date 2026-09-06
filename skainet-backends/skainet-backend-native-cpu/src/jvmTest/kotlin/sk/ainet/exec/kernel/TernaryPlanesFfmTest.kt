package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.TernaryPlanesMatmulKernel
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1150 end-to-end on the JVM: the REAL vendored `hs_ml_lmhead_stage1` (via
 * `skainet_ternary_lmhead_stage1`, FFM) behind the REAL dispatcher, against the full 8-plane
 * Kotlin reference. The C kernel spawns its 4 pthreads at any output_dim, so every case here
 * also exercises its internal threading.
 */
class TernaryPlanesFfmTest {

    @BeforeTest fun setUp() {
        KernelDispatch.clearForTesting()
        assertTrue(NativeTernaryLmheadKernel.isAvailable(), "bundled libskainet_kernels must resolve")
    }
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    private fun assertDispatchParity(n: Int, k: Int, seed: Int) {
        assertEquals("ternary_planes_matmul/ffm", NativeTernaryLmheadKernel.install())

        val rng = Random(seed)
        val values = FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f }
        val w = BitNetPlanesTensorData.fromFloats(Shape(n, k), values).packedView
        val a = TensorView.dense(
            Storage.Heap.wrap(FloatArray(k) { rng.nextFloat() - 0.5f }),
            Shape(1, k), FP32,
        )
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
        assertEquals("ternary_planes_matmul/ffm", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)

        val fromReference = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        TernaryPlanesMatmulKernel(TernaryPlanesMatmulKernel.keyFor()).run(listOf(a, w), fromReference)
        for (o in 0 until n) {
            val got = out.get(0, o)
            val want = fromReference.get(0, o)
            assertTrue(
                abs(got - want) <= 1e-3f * maxOf(1f, abs(want)),
                "[$o]: ffm=$got reference=$want (n=$n k=$k)",
            )
        }
    }

    @Test fun small_head() = assertDispatchParity(n = 8, k = 64, seed = 1)

    @Test fun bitnet_hidden_size_vocab_slice() = assertDispatchParity(n = 512, k = 2560, seed = 2)
}
