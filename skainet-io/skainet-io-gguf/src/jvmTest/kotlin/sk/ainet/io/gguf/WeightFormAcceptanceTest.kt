package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.KernelCapabilities
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightFormResolver
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.memory.plan.resolveWeightForms
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.math.abs
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * #1118, the acceptance criterion for #1109: **the model author writes nothing about weight forms,
 * and the same code is correct on two very different devices.**
 *
 * Everything else in #1109 is machinery for this one claim. So the test is arranged to make the
 * claim falsifiable rather than to exercise the machinery: [userCode] below is written once, takes
 * no policy, no profile and no form, and is called identically on both paths. If honouring a device
 * ever required the caller to say something, this file would have to change to keep passing.
 */
class WeightFormAcceptanceTest {

    @BeforeTest
    fun registerKernels() {
        // What every PlatformCpuOpsFactory does at startup, and what this module does not get for
        // free: consumed as a dependency, `DirectCpuExecutionContext` resolves to the *common*
        // `DefaultCpuOps` with an empty registry, and packed matmul is silently wrong in exactly
        // that configuration (#1124, found by this test). Registering the scalar provider puts the
        // test in the configuration a real application runs in.
        KernelRegistry.register(ScalarKernelProvider)
    }

    // Three Q8_0 blocks per row, so block order is discriminable (#968), and an output dimension
    // that is a whole number of blocks, so the relayouted weight is row-block-aligned.
    private val outDim = 32
    private val inDim = 96

    /**
     * The unchanged snippet. No profile, no form, no policy — a weight and an activation.
     *
     * This is the whole point of #1109: whether `w` arrived packed, dequantized, heap or mapped is
     * decided elsewhere, and none of it appears here.
     */
    private fun userCode(x: Tensor<FP32, Float>, w: Tensor<FP32, Float>): FloatArray =
        x.matmulWeightTransposed(w).data.copyToFloatArray()

    private fun modelFile(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("blk.0.attn_q.weight", GGMLQuantizationType.Q8_0, elements = outDim * inDim)
            .copy(dims = listOf(inDim.toLong(), outDim.toLong())),
    )

    /** Load [f] the way a device described by [profile] and [capabilities] calls for. */
    private fun loadFor(
        f: File,
        profile: PlannerProfile,
        capabilities: KernelCapabilities,
    ): Pair<WeightForm, Tensor<FP32, Float>> {
        // The device-dependent axes come from the resolver. The shape axis does not, by design:
        // which way round a checkpoint labels its dimensions is a property of the *format*, not of
        // the machine, so it is stated once here and is identical on both paths. GGUF writes `ne`
        // order, the engine means [out, in].
        val form = WeightFormResolver.resolve(TensorEncoding.Q8_0, profile, capabilities)
            .copy(shape = WeightShapeOrientation.OUT_IN)
        val ctx = DirectCpuExecutionContext()
        var weight: Tensor<FP32, Float>? = null
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                weightForm = form,
            ).load<FP32, Float>(ctx, FP32::class) { _, tensor -> weight = tensor }
        }
        return form to weight!!
    }

    private fun activation(): Tensor<FP32, Float> {
        val ctx = DirectCpuExecutionContext()
        return ctx.fromFloatArray<FP32, Float>(
            Shape(1, inDim), FP32::class, FloatArray(inDim) { (it % 13) * 0.0625f },
        )
    }

    @Test
    fun `one model and one snippet of code run correctly on a desktop and on a 2 GB board`() {
        val f = modelFile()
        try {
            // A desktop build without a Q8_0 kernel: it has the memory to absorb a widening, so
            // the resolver dequantizes once at load rather than on every forward pass.
            val (desktopForm, desktopWeight) = loadFor(f, PlannerProfile.DESKTOP, KernelCapabilities.DENSE_ONLY)
            // A 2 GB board with the packed kernels SKaiNET ships: weights mapped, encoding kept.
            val (mobileForm, mobileWeight) = loadFor(f, PlannerProfile.MOBILE_2GB, KernelCapabilities.EVERYTHING)

            // 1. The two devices resolved to different forms — asserted, not assumed.
            assertNotEquals(desktopForm, mobileForm, "if both devices got the same form this test proves nothing")
            assertEquals(
                EncodingRequest.DequantizeTo(FP32), desktopForm.encoding,
                "no kernel for Q8_0 here, and a desktop can afford to pay once at load",
            )
            assertEquals(EncodingRequest.KeepAsStored, mobileForm.encoding, "the board can feed Q8_0, so it keeps it")
            assertEquals(WeightResidency.HEAP, desktopForm.residency)
            assertEquals(WeightResidency.MAPPED, mobileForm.residency, "a 2 GB board maps its weights")

            // 2. And the same code, unchanged, is correct on both.
            val x = activation()
            val desktopOut = userCode(x, desktopWeight)
            val mobileOut = userCode(x, mobileWeight)

            assertEquals(desktopOut.size, mobileOut.size)
            for (o in desktopOut.indices) {
                // Not bit-identical, and should not be claimed to be: one path multiplies through a
                // Q8_0 kernel and the other through dequantized floats, so they differ by
                // quantization error, not by disagreement about the matrix.
                val tolerance = 1e-2f * maxOf(1.0f, abs(desktopOut[o]))
                assertTrue(
                    abs(desktopOut[o] - mobileOut[o]) <= tolerance,
                    "output[$o]: desktop ${desktopOut[o]} vs mobile ${mobileOut[o]}",
                )
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the mobile plan knows what its form costs before the load happens`() {
        val f = modelFile()
        try {
            val stored = JvmRandomAccessSource.open(f).use { src ->
                StreamingGGUFReader.open(src).planInput(ctx = 512)
            }

            val kept = stored.resolveWeightForms(PlannerProfile.MOBILE_2GB, KernelCapabilities.EVERYTHING)
            // Strict is the point of MOBILE_2GB, so pricing a widening on it means opting out of
            // the refusal first — which is exactly the decision the number is meant to inform.
            val lenientMobile = PlannerProfile.MOBILE_2GB.copy(strict = false)
            val dequantized = stored.resolveWeightForms(lenientMobile, KernelCapabilities.DENSE_ONLY)

            val keptPlan = MemoryPlans.plan(kept)
            val dequantizedPlan = MemoryPlans.plan(dequantized)

            assertEquals(0L, keptPlan.formConversionBytes, "keeping the stored encoding converts nothing")
            assertTrue(
                dequantizedPlan.formConversionBytes > 0,
                "dequantizing costs something and the plan must say so before the load, not after",
            )
            assertEquals(
                dequantizedPlan.weightsBytes - keptPlan.weightsBytes,
                dequantizedPlan.formConversionBytes,
                "and the number it reports is exactly the difference between the two plans",
            )
        } finally {
            f.delete()
        }
    }

    @Test
    fun `a strict board is told about a missing kernel instead of quietly paying for it`() {
        // MOBILE_2GB's own documentation calls dispatcher-inserted dequantization "the defect it
        // is". With strict set, the resolver refuses rather than resolving to a 4x load.
        val failure = kotlin.runCatching {
            WeightFormResolver.resolve(TensorEncoding.Q8_0, PlannerProfile.MOBILE_2GB, KernelCapabilities.DENSE_ONLY)
        }.exceptionOrNull()

        assertTrue(failure is IllegalStateException, "MOBILE_2GB is strict by default, so this must refuse: $failure")
        assertTrue(failure.message!!.contains("Q8_0"), failure.message!!)

        // A build that would rather load slowly than not at all opts out explicitly.
        val lenient = WeightFormResolver.resolve(
            TensorEncoding.Q8_0, PlannerProfile.MOBILE_2GB.copy(strict = false), KernelCapabilities.DENSE_ONLY,
        )
        assertEquals(EncodingRequest.DequantizeTo(FP32), lenient.encoding)
    }
}
