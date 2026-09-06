package sk.ainet.lang.memory.plan

import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1109 slice 1: the form is decided from the file, the device and the kernels — not by the caller.
 *
 * The table below is the specification. Every encoding is resolved with a kernel present and with
 * one absent, under both a desktop and a 2 GB profile, and each cell states what should come out
 * and why. If a rule changes, this table is where it changes.
 */
class WeightFormResolverTest {

    private val packed: List<TensorEncoding> = listOf(
        TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0,
        TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
        TensorEncoding.TQ1_0, TensorEncoding.TQ2_0,
    )

    /** A target that can feed exactly [supported] and nothing else. */
    private fun capableOf(vararg supported: TensorEncoding): KernelCapabilities =
        object : KernelCapabilities {
            override fun canFeedMatmul(encoding: TensorEncoding?): Boolean =
                encoding == null || encoding in supported
        }

    @Test
    fun `a weight whose kernel exists keeps its encoding and gets feed order when it can be produced`() {
        for (encoding in packed) {
            for (profile in listOf(PlannerProfile.DESKTOP, PlannerProfile.MOBILE_2GB)) {
                val form = WeightFormResolver.resolve(
                    encoding, profile, capableOf(encoding), canProduceKernelFeedOrder = true,
                )
                assertEquals(
                    EncodingRequest.KeepAsStored, form.encoding,
                    "${encoding.name} on ${profile.name}: a feedable weight must not be re-encoded",
                )
                assertEquals(
                    WeightByteOrder.KERNEL_FEED, form.order,
                    "${encoding.name} on ${profile.name}: the kernel reads input-block-major, so load it that way",
                )
            }
        }
    }

    @Test
    fun `feed order is not asked for by default because nothing can produce it yet`() {
        // The gap #1118's end-to-end test found: the resolver asked for KERNEL_FEED and the loader
        // had to reject it, for the *common* case of a target that has kernels. Wanting feed order
        // is a fact about the kernel; producing it is a fact about the loader, and until #1120 the
        // answer to the second is no. Neither slice's own tests could see this — only running them
        // together could.
        for (encoding in packed) {
            val form = WeightFormResolver.resolve(encoding, PlannerProfile.DESKTOP, capableOf(encoding))
            assertEquals(
                WeightByteOrder.AS_STORED, form.order,
                "${encoding.name}: the default resolution must be one a loader can actually honour",
            )
        }
    }

    @Test
    fun `a weight with no kernel is dequantized once at load rather than every forward pass`() {
        for (encoding in packed) {
            val form = WeightFormResolver.resolve(encoding, PlannerProfile.DESKTOP, KernelCapabilities.DENSE_ONLY)
            assertEquals(
                EncodingRequest.DequantizeTo(FP32), form.encoding,
                "${encoding.name}: with no kernel the backend would dequantize per call; once is better",
            )
            assertEquals(WeightByteOrder.AS_STORED, form.order, "${encoding.name}: dense bytes have no block order")
        }
    }

    @Test
    fun `a strict profile refuses the silent dequantization instead of paying for it`() {
        val strict = PlannerProfile.MOBILE_2GB.copy(strict = true)
        val failure = assertFailsWith<IllegalStateException> {
            WeightFormResolver.resolve(TensorEncoding.Q4_K, strict, KernelCapabilities.DENSE_ONLY)
        }
        val message = failure.message!!
        assertTrue(message.contains("Q4_K"), "it names the encoding: $message")
        assertTrue(message.contains("strict"), "and why it refused: $message")
        assertTrue(message.contains("×"), "and what it would have cost: $message")
    }

    @Test
    fun `residency follows the device and nothing else`() {
        val mapped = PlannerProfile.DESKTOP.copy(weightsMapped = true)
        for (encoding in packed + listOf(null)) {
            assertEquals(
                WeightResidency.MAPPED,
                WeightFormResolver.resolve(encoding, mapped, KernelCapabilities.EVERYTHING).residency,
                "${encoding?.name ?: "dense"}: weightsMapped is a statement about the device, not the encoding",
            )
            assertEquals(
                WeightResidency.HEAP,
                WeightFormResolver.resolve(encoding, PlannerProfile.DESKTOP, KernelCapabilities.EVERYTHING).residency,
            )
        }
    }

    @Test
    fun `a dense weight asks for nothing`() {
        val form = WeightFormResolver.resolve(null, PlannerProfile.DESKTOP, KernelCapabilities.DENSE_ONLY)
        assertTrue(form.isPassThrough, "a dense weight on a plain profile is already in its only form: $form")
        assertEquals(WeightForm.AS_STORED_ON_HEAP, form)
    }

    @Test
    fun `the same weight resolves differently on two targets`() {
        // The point of the whole exercise, in one assertion: identical input, different answers,
        // and the model author wrote nothing either way.
        val withKernel = WeightFormResolver.resolve(
            TensorEncoding.Q4_K, PlannerProfile.MOBILE_2GB.copy(weightsMapped = true), capableOf(TensorEncoding.Q4_K),
        )
        val withoutKernel = WeightFormResolver.resolve(
            TensorEncoding.Q4_K, PlannerProfile.DESKTOP, KernelCapabilities.DENSE_ONLY,
        )
        assertTrue(withKernel != withoutKernel, "same file, different devices, same resolved form — that would be the bug")
        assertEquals(EncodingRequest.KeepAsStored, withKernel.encoding)
        assertEquals(EncodingRequest.DequantizeTo(FP32), withoutKernel.encoding)
    }

    @Test
    fun `a backend whose packed kernels read canonical order is not handed a pointless permutation`() {
        val canonicalReader = object : KernelCapabilities {
            override fun canFeedMatmul(encoding: TensorEncoding?): Boolean = true
            override fun wantsKernelFeedOrder(encoding: TensorEncoding): Boolean = false
        }
        val form = WeightFormResolver.resolve(TensorEncoding.Q8_0, PlannerProfile.DESKTOP, canonicalReader)
        assertEquals(WeightByteOrder.AS_STORED, form.order, "relayouting for a kernel that does not want it is waste")
    }
}
