package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Owner
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.StorageId
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1138: the exact FP32×`BITNET_B1_58` path takes over dispatch when its native kernel is present,
 * and its absence changes nothing — FP32 activations keep flowing through the int8-requantize →
 * `bitnet_gemv` path exactly as before, with a notice instead of a crash.
 */
class TernaryF32KernelPackTest {

    private val k = 64
    private val n = 4

    @BeforeTest fun setUp() = KernelDispatch.clearForTesting()
    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    /** A stand-in for the FFM/JNI kernel: records calls, computes the unscaled codes-dot. */
    private class FakeNative(override val name: String = "fake-lut") : TernaryF32GemvNative {
        var calls: Int = 0
        override fun gemvPacked(
            activation: FloatArray, activationOffset: Int,
            weight: ByteArray, weightByteOffset: Int,
            inputDim: Int, outputDim: Int,
            out: FloatArray, outOffset: Int,
        ) {
            calls++
            for (o in 0 until outputDim) {
                var acc = 0f
                for (i in 0 until inputDim) {
                    val element = o * inputDim + i
                    val code = ((weight[weightByteOffset + element / 4].toInt()
                        shr ((element % 4) * 2)) and 3) - 1
                    acc += code * activation[activationOffset + i]
                }
                out[outOffset + o] = acc
            }
        }
    }

    /**
     * A minimal non-Heap [Storage] usable from `commonTest` (no platform off-heap type is
     * constructible here) — enough to exercise [NativeTernaryF32ViewKernel]'s `is Storage.OffHeap`
     * branch and its [TernaryF32GemvNative.gemvPackedStorage] dispatch without a real JVM/Native
     * off-heap allocation. The zero-copy behavior itself (a real `MemorySegment`) is pinned by
     * `NativeTernaryF32GemvKernelTest` in skainet-backend-native-cpu.
     */
    private class FakeOffHeapStorage(private val bytes: ByteArray) : Storage.OffHeap() {
        override val id: StorageId = StorageId.next()
        override val sizeBytes: Long get() = bytes.size.toLong()
        override val owner: Owner = Owner.Owned(ScopeKind.AMBIENT)
        override val debugOrigin: TensorId? = null
        override val sink: TraceSink = NoopTraceSink
        override val isMutable: Boolean get() = true

        override fun slice(offsetBytes: Long, lengthBytes: Long): Storage =
            FakeOffHeapStorage(bytes.copyOfRange(offsetBytes.toInt(), (offsetBytes + lengthBytes).toInt()))

        override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
            bytes.copyInto(dest, destOffset, offset.toInt(), offset.toInt() + length)
        }

        override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
            src.copyInto(bytes, offset.toInt(), srcOffset, srcOffset + length)
        }
    }

    private fun weight(): TensorView {
        var seed = 5
        val values = FloatArray(n * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val bytes = TernaryCodec.encodeBitNet(values)
        return TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(n, k), TensorEncoding.BITNET_B1_58,
            TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * k),
        )
    }

    private fun activation(rows: Int = 1): TensorView {
        var seed = 9
        val floats = FloatArray(rows * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 2000 - 1000) / 1000f
        }
        return TensorView.dense(Storage.Heap.wrap(floats), Shape(rows, k), FP32)
    }

    @Test
    fun withoutTheArtifactTheInt8PathServesUnchangedAndTheCallerIsTold() {
        TernaryKernelPacks.install(native = null, warn = {})

        val warnings = mutableListOf<String>()
        val serving = TernaryF32KernelPack.install(native = null, warn = { warnings += it })
        assertEquals(TernaryF32KernelPack.NOT_INSTALLED, serving)
        assertEquals(1, warnings.size, "exactly one notice, not a crash: $warnings")
        assertTrue(warnings.single().contains("int8-requantize"), warnings.single())

        // FP32 × b1.58 dispatch behaves exactly as before this pack existed:
        // requantize adapter + the int8 reference.
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(activation(), weight(), out, Scope.Ambient, sink)
        assertEquals("bitnet_gemv/reference", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
    }

    @Test
    fun withTheArtifactTheExactKeyBeatsTheRequantizePath() {
        TernaryKernelPacks.install(native = null, warn = {})
        val native = FakeNative()
        val serving = TernaryF32KernelPack.install(native)
        assertEquals("ternary_f32_gemv/fake-lut", serving)

        val w = weight()
        val a = activation()
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)

        assertEquals("ternary_f32_gemv/fake-lut", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
        assertEquals(1, native.calls)
        assertTrue(
            sink.eventsOf<TraceEvent.AdapterInserted>().isEmpty(),
            "the exact f32 path needs no requantize adapter",
        )

        // and the scaled result equals the f32 reference
        val fromReference = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), fromReference)
        for (o in 0 until n) {
            val got = out.get(0, o)
            val want = fromReference.get(0, o)
            assertTrue(abs(got - want) <= 1e-5f * maxOf(1f, abs(want)), "[$o]: $got vs $want")
        }
    }

    @Test
    fun multiRowActivationsLoopTheNativeGemvPerRow() {
        val native = FakeNative()
        TernaryF32KernelPack.install(native)
        val rows = 3
        val w = weight()
        val a = activation(rows)
        val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        NativeTernaryF32ViewKernel(native, TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)
        assertEquals(rows, native.calls, "prefill loops the gemv, one call per row")

        val fromReference = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), fromReference)
        for (r in 0 until rows) for (o in 0 until n) {
            val got = out.get(r, o)
            val want = fromReference.get(r, o)
            assertTrue(abs(got - want) <= 1e-5f * maxOf(1f, abs(want)), "[$r,$o]: $got vs $want")
        }
    }

    @Test
    fun offHeapWeightStorageStillTakesTheNativePathNotTheReferenceFallback() {
        // #1202: shipping off-heap ternary storage must not silently regress the native path to
        // the slow reference kernel — this is the direct regression test for that risk.
        val native = FakeNative()
        TernaryF32KernelPack.install(native)
        val rows = 2
        val a = activation(rows)

        var seed = 5
        val values = FloatArray(n * k) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1) * 0.5f
        }
        val bytes = TernaryCodec.encodeBitNet(values)
        val offHeapStorage = FakeOffHeapStorage(bytes)
        val w = TensorView.packed(
            offHeapStorage, Shape(n, k), TensorEncoding.BITNET_B1_58,
            TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * k),
        )

        val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        NativeTernaryF32ViewKernel(native, TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)
        assertEquals(rows, native.calls, "off-heap weight must still dispatch to the native kernel, once per row")

        val fromReference = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
        val heapWeight = TensorView.packed(
            Storage.Heap.wrap(bytes), Shape(n, k), TensorEncoding.BITNET_B1_58,
            TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * k),
        )
        TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, heapWeight), fromReference)
        for (r in 0 until rows) for (o in 0 until n) {
            val got = out.get(r, o)
            val want = fromReference.get(r, o)
            assertTrue(abs(got - want) <= 1e-5f * maxOf(1f, abs(want)), "[$r,$o]: $got vs $want")
        }
    }

    @Test
    fun kIndivisibleByFourFallsBackToTheReferenceInsteadOfFailing() {
        val native = FakeNative()
        val oddK = 6
        var seed = 3
        val values = FloatArray(n * oddK) {
            seed = seed * 1103515245 + 12345
            ((seed ushr 16) % 3 - 1).toFloat()
        }
        val w = TensorView.packed(
            Storage.Heap.wrap(TernaryCodec.encodeBitNet(values)), Shape(n, oddK),
            TensorEncoding.BITNET_B1_58, TernaryBlockDecoder(TensorEncoding.BITNET_B1_58, n * oddK),
        )
        val floats = FloatArray(oddK) { (it + 1).toFloat() }
        val a = TensorView.dense(Storage.Heap.wrap(floats), Shape(1, oddK), FP32)
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        NativeTernaryF32ViewKernel(native, TernaryF32GemvKernel.keyFor()).run(listOf(a, w), out)
        assertEquals(0, native.calls, "the packing crosses byte boundaries — reference serves")

        val decoded = TernaryCodec.decodeBitNet((w.storage as Storage.Heap).bytes!!, n * oddK)
        for (o in 0 until n) {
            var want = 0f
            for (i in 0 until oddK) want += floats[i] * decoded[o * oddK + i]
            assertTrue(abs(out.get(0, o) - want) <= 1e-5f, "[$o]: ${out.get(0, o)} vs $want")
        }
    }

    @Test
    fun theCapabilityIsRecordedInTheKey() {
        TernaryF32KernelPack.install(FakeNative(), setOf("ffm"))
        val keys = KernelDispatch.kernels().filter { it.name.startsWith("ternary_f32_gemv/fake") }.map { it.key }
        assertTrue(keys.any { it.capabilities == setOf("ffm") }, "the pack declares what it needs: $keys")
        assertTrue(keys.any { it.capabilities.isEmpty() }, "and is reachable from an operand-only key: $keys")
    }
}
