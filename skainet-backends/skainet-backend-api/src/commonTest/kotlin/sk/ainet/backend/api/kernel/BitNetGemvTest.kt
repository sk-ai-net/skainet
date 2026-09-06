package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ForwardScope
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
 * #1040 (M2-F3): int8 activations against ternary weights, and the adapter that produces them.
 */
class BitNetGemvTest {

    private val k = 256          // one TQ block
    private val n = 4            // output rows

    @BeforeTest fun setUp() {
        KernelDispatch.clearForTesting()
        BitNetGemvKernel.registerReference()
    }

    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    /** Deterministic ternary weight values, `[n, k]`, scaled so the block absmax is exact in FP16. */
    private fun weightValues(seed: Int = 3): FloatArray {
        var s = seed
        return FloatArray(n * k) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 3 - 1) * 0.5f
        }
    }

    private fun activationValues(rows: Int, seed: Int = 11): FloatArray {
        var s = seed
        return FloatArray(rows * k) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2000 - 1000) / 1000f
        }
    }

    private fun weightView(encoding: TensorEncoding, values: FloatArray): TensorView {
        val bytes = TernaryCodec.encode(encoding, values)
        val decoder = if (encoding == TensorEncoding.BITNET_B1_58) {
            TernaryBlockDecoder(encoding, values.size)   // one scale for the whole tensor
        } else {
            TernaryBlockDecoder(encoding)
        }
        return TensorView.packed(Storage.Heap.wrap(bytes), Shape(n, k), encoding, decoder)
    }

    /** `out[r, o] = Σ_k decoded_activation[r, k] * decoded_weight[o, k]` — the definition. */
    private fun reference(activation: TensorView, weight: TensorView, rows: Int): FloatArray {
        val out = FloatArray(rows * n)
        for (r in 0 until rows) {
            for (o in 0 until n) {
                var acc = 0f
                for (i in 0 until k) acc += I8Absmax.valueAt(activation, r, i) * weight.get(o, i)
                out[r * n + o] = acc
            }
        }
        return out
    }

    // --- the adapter ---------------------------------------------------------------------------

    @Test
    fun requantizationKeepsTheValuesAndPricesItself() {
        val rows = 2
        val values = activationValues(rows)
        val dense = TensorView.dense(Storage.Heap.wrap(values), Shape(rows, k), FP32)
        val sink = RecordingTraceSink()
        val quantized = I8Absmax.requantize(dense, Scope.Ambient, sink)

        assertEquals(I8Absmax.FORMAT, quantized.format)
        for (r in 0 until rows) {
            var amax = 0f
            for (c in 0 until k) amax = maxOf(amax, abs(values[r * k + c]))
            // Kotlin/JS computes Float arithmetic in double precision, so the same division can
            // differ in the last bit between the test and the implementation: compare relatively.
            val expectedScale = amax / 127f
            assertTrue(
                abs(expectedScale - I8Absmax.scaleOf(quantized, r)) <= 1e-6f * expectedScale,
                "row $r scale should be absmax / 127 = $expectedScale, was ${I8Absmax.scaleOf(quantized, r)}",
            )
            val tolerance = I8Absmax.scaleOf(quantized, r)
            for (c in 0 until k) {
                assertTrue(
                    abs(values[r * k + c] - I8Absmax.valueAt(quantized, r, c)) <= tolerance,
                    "row $r col $c: ${values[r * k + c]} vs ${I8Absmax.valueAt(quantized, r, c)}",
                )
            }
        }

        val adapter = sink.eventsOf<TraceEvent.AdapterInserted>().single()
        assertEquals("requantize-i8-absmax", adapter.kind)
        assertEquals(I8Absmax.bytesFor(rows, k), adapter.bytes, "codes plus one scale per row")
        assertEquals(rows.toLong() * k + rows * 4, adapter.bytes)
    }

    @Test
    fun aZeroRowSurvivesQuantization() {
        val dense = TensorView.dense(Storage.Heap.wrap(FloatArray(k)), Shape(1, k), FP32)
        val quantized = I8Absmax.requantize(dense, Scope.Ambient)
        assertEquals(0f, I8Absmax.scaleOf(quantized, 0), "no division by zero")
        for (c in 0 until k) assertEquals(0f, I8Absmax.valueAt(quantized, row = 0, col = c))
    }

    @Test
    fun theAdapterCostPerStepIsTheOneTheDesignPredicts() {
        // §5.3 quotes ≈ 4 KB per decode step for a 2 B-parameter model (hidden 4096, one token).
        val bytes = I8Absmax.bytesFor(rows = 1, cols = 4096)
        assertEquals(4096L + 4, bytes)
        assertTrue(bytes < 5 * 1024, "one token's activations must stay in the kilobytes: $bytes")
    }

    // --- the kernel ----------------------------------------------------------------------------

    @Test
    fun theReferenceKernelMatchesTheDefinitionForEveryTernaryEncoding() {
        for (encoding in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0, TensorEncoding.BITNET_B1_58)) {
            val rows = 2
            val weight = weightView(encoding, weightValues())
            val activation = I8Absmax.requantize(
                TensorView.dense(Storage.Heap.wrap(activationValues(rows)), Shape(rows, k), FP32),
                Scope.Ambient,
            )
            val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
            BitNetGemvKernel(BitNetGemvKernel.keyFor(weight.format)).run(listOf(activation, weight), out)

            val expected = reference(activation, weight, rows)
            for (r in 0 until rows) for (o in 0 until n) {
                val got = out.get(r, o)
                val want = expected[r * n + o]
                assertTrue(
                    abs(got - want) <= 1e-3f * maxOf(1f, abs(want)),
                    "${encoding.name} [$r,$o]: $got vs $want",
                )
            }
        }
    }

    @Test
    fun zeroWeightsContributeNothing() {
        // every weight zero → every output zero, whatever the activations are
        val encoding = TensorEncoding.TQ2_0
        val weight = weightView(encoding, FloatArray(n * k))
        val activation = I8Absmax.requantize(
            TensorView.dense(Storage.Heap.wrap(activationValues(1)), Shape(1, k), FP32),
            Scope.Ambient,
        )
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        BitNetGemvKernel(BitNetGemvKernel.keyFor(weight.format)).run(listOf(activation, weight), out)
        for (o in 0 until n) assertEquals(0f, out.get(0, o))
    }

    // --- dispatch ------------------------------------------------------------------------------

    @Test
    fun theDispatcherRequantizesTheActivationAndPicksTheTernaryKernel() {
        val encoding = TensorEncoding.TQ2_0
        val weight = weightView(encoding, weightValues())
        val floats = activationValues(1)
        val activation = TensorView.dense(Storage.Heap.wrap(floats), Shape(1, k), FP32)
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)

        val sink = RecordingTraceSink()
        val scope = ForwardScope(slabFloats = 4 * k, sink = sink, name = "decode")
        KernelDispatch.matmul(activation, weight, out, scope, sink)

        val kernels = sink.eventsOf<TraceEvent.KernelRun>()
        assertEquals(1, kernels.size)
        assertEquals("bitnet_gemv/reference", kernels.single().kernel, "a ternary weight selects the ternary kernel")
        val adapter = sink.eventsOf<TraceEvent.AdapterInserted>().single()
        assertEquals("requantize-i8-absmax", adapter.kind, "and the activation adapter is visible, not hidden")
        assertEquals(I8Absmax.FORMAT, adapter.to)

        // the numbers are the kernel's own
        val quantized = I8Absmax.requantize(activation, Scope.Ambient)
        val expected = reference(quantized, weight, rows = 1)
        for (o in 0 until n) assertTrue(abs(out.get(0, o) - expected[o]) <= 1e-3f * maxOf(1f, abs(expected[o])), "[$o]")
        scope.close()
    }

    @Test
    fun aDenseWeightIsUntouchedByAnyOfThis() {
        val weightFloats = FloatArray(n * k) { (it % 5) * 0.25f }
        val weight = TensorView.dense(Storage.Heap.wrap(weightFloats), Shape(n, k), FP32)
        val activation = TensorView.dense(Storage.Heap.wrap(activationValues(1)), Shape(1, k), FP32)
        val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(activation, weight, out, Scope.Ambient, sink)
        assertTrue(
            sink.eventsOf<TraceEvent.AdapterInserted>().none { it.kind == "requantize-i8-absmax" },
            "only ternary formats ask for int8 activations",
        )
    }
}
