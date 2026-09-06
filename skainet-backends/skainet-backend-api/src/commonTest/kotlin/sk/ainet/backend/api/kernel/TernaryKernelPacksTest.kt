package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.I8Absmax
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
 * #1041 (M2-F4): a platform `bitnet_gemv` takes over when its artifact is present, and its absence
 * is a warning and a slower kernel — never a crash.
 *
 * The NEON kernel itself is C and is tested where it can run; what belongs here is the contract
 * around it, which every target can check: who gets registered, what happens without the artifact,
 * and that the native path is only taken for the shapes it declares.
 */
class TernaryKernelPacksTest {

    private val k = 256
    private val n = 4

    @BeforeTest fun setUp() = KernelDispatch.clearForTesting()
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    /** A stand-in for the JNI kernel: records that it ran, and computes the same thing. */
    private class FakeNative(override val name: String = "neon-dotprod") : BitNetGemvNative {
        var calls: Int = 0
        override fun gemvTq2_0(
            activation: ByteArray, activationOffset: Int, activationScale: Float,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            out: FloatArray, outOffset: Int,
        ) {
            calls++
            val codes = TernaryCodec.codes(TensorEncoding.TQ2_0, weight, outputDim * inputDim, weightByteOffset)
            val blocks = inputDim / 256
            for (o in 0 until outputDim) {
                var acc = 0f
                for (b in 0 until blocks) {
                    val scaleOffset = weightByteOffset + ((o * blocks + b) * 66) + 64
                    val d = sk.ainet.lang.types.Fp16Codec.decode(
                        (weight[scaleOffset].toInt() and 0xFF) or ((weight[scaleOffset + 1].toInt() and 0xFF) shl 8),
                    )
                    var partial = 0
                    for (i in 0 until 256) {
                        val code = codes[o * inputDim + b * 256 + i].toInt()
                        if (code != 0) {
                            val a = activation[activationOffset + b * 256 + i].toInt()
                            partial += if (code > 0) a else -a
                        }
                    }
                    acc += partial * d
                }
                out[outOffset + o] = acc * activationScale
            }
        }
    }

    private fun weight(): TensorView {
        var seed = 5
        val values = FloatArray(n * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val bytes = TernaryCodec.encode(TensorEncoding.TQ2_0, values)
        return TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(n, k), TensorEncoding.TQ2_0,
            TernaryBlockDecoder(TensorEncoding.TQ2_0),
        )
    }

    private fun activation(rows: Int = 1): TensorView {
        var seed = 9
        val floats = FloatArray(rows * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 2000 - 1000) / 1000f
        }
        return I8Absmax.requantize(TensorView.dense(Storage.Heap.wrap(floats), Shape(rows, k), FP32), Scope.Ambient)
    }

    @Test
    fun withoutTheArtifactTheReferenceServesAndTheCallerIsTold() {
        val warnings = mutableListOf<String>()
        val serving = TernaryKernelPacks.install(native = null, warn = { warnings += it })

        assertEquals("bitnet_gemv/reference", serving)
        assertEquals(1, warnings.size, "exactly one notice, not a crash: $warnings")
        assertTrue(warnings.single().contains("portable reference"), warnings.single())

        // and dispatch still works
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(activation(), weight(), out, Scope.Ambient, sink)
        assertEquals("bitnet_gemv/reference", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
    }

    @Test
    fun withTheArtifactTheNativeKernelTakesOverAndAgreesWithTheReference() {
        val native = FakeNative()
        val warnings = mutableListOf<String>()
        val serving = TernaryKernelPacks.install(native, setOf(TernaryKernelPacks.CAPABILITY_DOTPROD)) { warnings += it }
        assertEquals("bitnet_gemv/neon-dotprod", serving)
        assertTrue(warnings.isEmpty(), "nothing to warn about: $warnings")

        val w = weight()
        val a = activation()
        val fromNative = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, fromNative, Scope.Ambient, sink)
        assertEquals("bitnet_gemv/neon-dotprod", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
        assertEquals(1, native.calls)

        val fromReference = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        BitNetGemvKernel(BitNetGemvKernel.keyFor(w.format)).run(listOf(a, w), fromReference)
        for (o in 0 until n) {
            val got = fromNative.get(0, o)
            val want = fromReference.get(0, o)
            assertTrue(abs(got - want) <= 1e-5f * maxOf(1f, abs(want)), "[$o]: $got vs $want")
        }
    }

    @Test
    fun theNativePathIsSkippedForShapesItDoesNotTake() {
        val native = FakeNative()
        TernaryKernelPacks.install(native)
        val w = weight()
        // prefill: more than one row is not what the gemv takes — the reference runs instead
        val out = TensorView.dense(Storage.Heap.floats(2 * n), Shape(2, n), FP32)
        NativeBitNetGemvKernel(native, BitNetGemvKernel.keyFor(w.format)).run(listOf(activation(rows = 2), w), out)
        assertEquals(0, native.calls, "a multi-row activation falls back rather than failing")
        var nonZero = false
        for (r in 0 until 2) for (o in 0 until n) if (out.get(r, o) != 0f) nonZero = true
        assertTrue(nonZero, "and it still computed the answer")
    }

    @Test
    fun theCapabilityIsRecordedInTheKey() {
        val native = FakeNative()
        TernaryKernelPacks.install(native, setOf(TernaryKernelPacks.CAPABILITY_DOTPROD))
        val keys = KernelDispatch.kernels().filter { it.name.startsWith("bitnet_gemv/neon") }.map { it.key }
        assertTrue(
            keys.any { it.capabilities == setOf(TernaryKernelPacks.CAPABILITY_DOTPROD) },
            "the pack declares what it needs: $keys",
        )
        assertTrue(keys.any { it.capabilities.isEmpty() }, "and is reachable from an operand-only key: $keys")
    }
}
