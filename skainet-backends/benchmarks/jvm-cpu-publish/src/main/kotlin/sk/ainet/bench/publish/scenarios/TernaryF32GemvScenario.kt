package sk.ainet.bench.publish.scenarios

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.TernaryF32GemvKernel
import sk.ainet.backend.api.kernel.TernaryKernelPacks
import sk.ainet.bench.publish.runner.Scenario
import sk.ainet.exec.kernel.NativeTernaryF32GemvKernel
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.types.FP32
import kotlin.random.Random

/**
 * FP32 activation × `BITNET_B1_58` ternary weight through the REAL dispatch path (#1141) — the
 * first ternary scenario, and deliberately not a raw kernel loop: what is measured is what a
 * decode step pays, adapters included.
 *
 * Providers select the path `KernelDispatch.matmul` takes for the same operands:
 *
 * - `ffm-lut` — the vendored NeoGPU LUT kernel (#1137) behind the exact FP32×b1.58 key (#1138):
 *   exact math, no requantization step. The kernel threads internally above 512 output rows.
 * - `int8` — the pre-existing W1.58A8 path: per-call I8-absmax requantization adapter + the
 *   portable `bitnet_gemv` reference (the JVM has no native `bitnet_gemv`), ~1.5 % quant error.
 * - `f32-reference` — the portable Kotlin f32 reference registered on the exact key: exact math
 *   at reference speed, the LUT kernel's correctness oracle.
 *
 * Dims default to the BitNet-2B FFN projection (k=2560, n=6912); the n=6912 regime also crosses
 * the LUT kernel's internal pthread threshold. Primary metric: GOP/s over `2·k·n` ops.
 */
internal class TernaryF32GemvScenario(
    smoke: Boolean,
    private val providerName: String,
) : Scenario {
    override val id: String = "engine-ternary-f32-gemv"
    override val suite: String = "skainet-engine"
    override val primaryMetric: String = "gops"
    override val unit: String = "gops"
    override val higherIsBetter: Boolean = true
    override val kernelProvider: String = providerName

    private val inputDim: Int = 2560
    private val outputDim: Int = if (smoke) 256 else 6912
    override val parameters: Map<String, String> = mapOf(
        "input_dim" to inputDim.toString(),
        "output_dim" to outputDim.toString(),
        "kernel" to providerName,
    )

    private lateinit var activation: TensorView
    private lateinit var weight: TensorView
    private lateinit var out: TensorView

    override fun setup() {
        KernelDispatch.clearForTesting()
        when (providerName.lowercase()) {
            "ffm-lut", "ffm", "native", "lut" -> {
                TernaryKernelPacks.install(native = null, warn = {})
                check(NativeTernaryF32GemvKernel.isAvailable()) {
                    "bundled libskainet_kernels missing — the ffm-lut provider cannot run"
                }
                NativeTernaryF32GemvKernel.install()
            }
            "int8", "requant" -> TernaryKernelPacks.install(native = null, warn = {})
            "f32-reference", "reference", "scalar" ->
                KernelDispatch.register(TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()))
            else -> error("unknown kernel provider: $providerName (use 'ffm-lut', 'int8' or 'f32-reference')")
        }

        val rng = Random((inputDim + outputDim).toLong())
        val values = FloatArray(outputDim * inputDim) { (rng.nextInt(3) - 1) * 0.5f }
        weight = BitNetB158TensorData.fromFloats(Shape(outputDim, inputDim), values).packedView
        activation = TensorView.dense(
            Storage.Heap.wrap(FloatArray(inputDim) { rng.nextFloat() - 0.5f }),
            Shape(1, inputDim), FP32,
        )
        out = TensorView.dense(Storage.Heap.floats(outputDim), Shape(1, outputDim), FP32)
    }

    override fun runOnce(): Double {
        val start = System.nanoTime()
        KernelDispatch.matmul(activation, weight, out, Scope.Ambient)
        val elapsedNs = System.nanoTime() - start
        @Suppress("UNUSED_VARIABLE") val sink = out.get(0, 0)
        val ops = 2.0 * inputDim.toDouble() * outputDim.toDouble()
        return (ops / (elapsedNs / 1_000_000_000.0)) / 1e9
    }

    override fun teardown() {
        KernelDispatch.clearForTesting()
    }
}
