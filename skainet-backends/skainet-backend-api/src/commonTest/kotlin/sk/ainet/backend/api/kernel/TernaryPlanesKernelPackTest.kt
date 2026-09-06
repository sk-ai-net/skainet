package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1150: the `BITNET_PLANES` pack contract. The native seam computes 4 fused planes per call;
 * the view kernel's two-call combination (`s0 + s4/81`) must equal the full 8-plane reference —
 * the dispatch invariant *matmul == decoded matmul* holds exactly.
 */
class TernaryPlanesKernelPackTest {

    private val k = 32
    private val n = 5

    @BeforeTest fun setUp() = KernelDispatch.clearForTesting()
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    /** Computes exactly the 4-plane fused contract, straight from the buffer layout. */
    private class FakeNative(override val name: String = "fake") : TernaryLmheadNative {
        var calls: Int = 0
        override fun lmheadStage1(
            activation: FloatArray, activationOffset: Int,
            weight: ByteArray, planesByteOffset: Int,
            planeStrideBytes: Int, rowScaleByteOffset: Int,
            inputDim: Int, outputDim: Int,
            out: FloatArray, outOffset: Int,
        ) {
            calls++
            val rowBytes = inputDim / 4
            for (o in 0 until outputDim) {
                val scaleBits = (weight[rowScaleByteOffset + o * 2].toInt() and 0xFF) or
                    ((weight[rowScaleByteOffset + o * 2 + 1].toInt() and 0xFF) shl 8)
                val scale = sk.ainet.lang.types.Fp16Codec.decode(scaleBits)
                var acc = 0f
                var w = 1f
                for (q in 0 until 4) {
                    val base = planesByteOffset + q * planeStrideBytes + o * rowBytes
                    var dot = 0f
                    for (i in 0 until inputDim) {
                        val code = ((weight[base + i / 4].toInt() and 0xFF) shr ((i % 4) * 2)) and 3
                        dot += (code - 1) * activation[activationOffset + i]
                    }
                    acc += dot * w
                    w /= 3f
                }
                out[outOffset + o] = acc * scale
            }
        }
    }

    private fun weight(): TensorView {
        val rng = Random(5)
        val values = FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f }
        return BitNetPlanesTensorData.fromFloats(Shape(n, k), values).packedView
    }

    private fun activation(rows: Int = 1): TensorView {
        val rng = Random(9)
        return TensorView.dense(
            Storage.Heap.wrap(FloatArray(rows * k) { rng.nextFloat() - 0.5f }),
            Shape(rows, k), FP32,
        )
    }

    @Test
    fun withTheArtifactTheExactKeyServesAndEqualsTheFullPlaneReference() {
        val native = FakeNative()
        val serving = TernaryPlanesKernelPack.install(native)
        assertEquals("ternary_planes_matmul/fake", serving)

        val w = weight()
        val a = activation()
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
        assertEquals("ternary_planes_matmul/fake", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
        assertEquals(2, native.calls, "full result = planes 0–3 + planes 4–7 / 81 — two fused calls")

        val fromReference = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        TernaryPlanesMatmulKernel(TernaryPlanesMatmulKernel.keyFor()).run(listOf(a, w), fromReference)
        for (o in 0 until n) {
            val got = out.get(0, o)
            val want = fromReference.get(0, o)
            assertTrue(abs(got - want) <= 1e-4f * maxOf(1f, abs(want)), "[$o]: $got vs $want")
        }
    }

    @Test
    fun withoutTheArtifactNothingIsRegisteredAndTheCallerIsTold() {
        val warnings = mutableListOf<String>()
        val serving = TernaryPlanesKernelPack.install(native = null, warn = { warnings += it })
        assertEquals(TernaryPlanesKernelPack.NOT_INSTALLED, serving)
        assertEquals(1, warnings.size)
        assertTrue(KernelDispatch.kernels().isEmpty(), "nothing registered without the native kernel")
    }

    @Test
    fun theReferenceEqualsTheDecodedMatmul() {
        val w = weight()
        val a = activation(rows = 2)
        val out = TensorView.dense(Storage.Heap.floats(2 * n), Shape(2, n), FP32)
        TernaryPlanesMatmulKernel(TernaryPlanesMatmulKernel.keyFor()).run(listOf(a, w), out)

        val bytes = (w.storage as Storage.Heap).bytes!!
        val decoded = TernaryCodec.decodeBitNetPlanes(bytes, n, k)
        for (r in 0 until 2) for (o in 0 until n) {
            var want = 0f
            for (i in 0 until k) want += a.get(r, i) * decoded[o * k + i]
            assertTrue(abs(out.get(r, o) - want) <= 1e-5f, "[$r,$o]: ${out.get(r, o)} vs $want")
        }
    }

    @Test
    fun encodingWithoutBlockSpecTriggersNoRequantizeAdapter() {
        // BITNET_PLANES deliberately declares no activation hint: without the pack the dispatcher
        // must fall to the decoding reference matmul, never the int8 requantize path.
        val w = weight()
        val a = activation()
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
        val kernel = sink.eventsOf<TraceEvent.KernelRun>().single().kernel
        assertEquals("reference", kernel, "decoding reference serves without the pack")
    }
}
