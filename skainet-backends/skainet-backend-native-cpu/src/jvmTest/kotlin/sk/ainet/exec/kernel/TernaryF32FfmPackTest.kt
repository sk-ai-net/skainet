package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.TernaryF32GemvKernel
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1138 end-to-end on the JVM: the REAL vendored kernel behind the REAL dispatcher. Install the
 * FFM pack, hand `KernelDispatch.matmul` an FP32 activation and a `BITNET_B1_58` weight view, and
 * the NeoGPU LUT kernel serves the exact key — no requantize adapter, results equal to the f32
 * reference.
 */
class TernaryF32FfmPackTest {

    private val k = 2560 // BitNet-2B hidden size
    private val n = 8

    @BeforeTest fun setUp() {
        KernelDispatch.clearForTesting()
        assertTrue(NativeTernaryF32GemvKernel.isAvailable(), "bundled libskainet_kernels must resolve")
    }
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    @Test
    fun theVendoredKernelServesDispatchAndMatchesTheReference() {
        val serving = NativeTernaryF32GemvKernel.install()
        assertEquals("ternary_f32_gemv/ffm", serving)

        var seed = 5
        val values = FloatArray(n * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val w = TensorView.packed(
            Storage.Heap.wrap(TernaryCodec.encodeBitNet(values)), Shape(n, k),
            TensorEncoding.BITNET_B1_58, TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * k),
        )
        seed = 9
        val a = TensorView.dense(
            Storage.Heap.wrap(FloatArray(k) { seed = seed * 1103515245 + 12345; ((seed ushr 16) % 2000 - 1000) / 1000f }),
            Shape(1, k), FP32,
        )

        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)

        assertEquals("ternary_f32_gemv/ffm", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
        assertTrue(sink.eventsOf<TraceEvent.AdapterInserted>().isEmpty(), "no requantize adapter on the exact path")

        val fromReference = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), fromReference)
        for (o in 0 until n) {
            val got = out.get(0, o)
            val want = fromReference.get(0, o)
            assertTrue(abs(got - want) <= 1e-3f * maxOf(1f, abs(want)), "[$o]: ffm=$got reference=$want")
        }
    }
}
