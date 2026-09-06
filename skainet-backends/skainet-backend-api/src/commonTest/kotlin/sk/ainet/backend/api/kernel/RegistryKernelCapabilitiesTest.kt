package sk.ainet.backend.api.kernel

import sk.ainet.lang.tensor.storage.TensorEncoding
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1109 slice 1: the capability answer comes from the registry, so it cannot disagree with dispatch.
 */
class RegistryKernelCapabilitiesTest {

    @AfterTest fun cleanup() { KernelDispatch.clearForTesting(); KernelRegistry.clearForTesting() }

    /** A provider that declares one packed matmul and, optionally, is not available here. */
    private class OneEncodingProvider(
        private val weightKey: String,
        private val available: Boolean = true,
    ) : KernelProvider {
        override val name: String = "fake-$weightKey"
        override val priority: Int = 100
        override fun isAvailable(): Boolean = available
        override fun matmulFp32(): Fp32MatmulKernel? = null
        override fun supports(opName: String, dtypeKeys: List<String>): Boolean =
            opName == "matmul" && dtypeKeys == listOf("Float32", weightKey)
    }

    @Test
    fun anEncodingWithARegisteredKernelCanBeFed() {
        KernelRegistry.clearForTesting()
        KernelRegistry.register(OneEncodingProvider("Q4_K"))

        assertTrue(RegistryKernelCapabilities.canFeedMatmul(TensorEncoding.Q4_K))
        assertFalse(
            RegistryKernelCapabilities.canFeedMatmul(TensorEncoding.Q6_K),
            "nothing registered a Q6_K kernel, so the honest answer is no",
        )
    }

    @Test
    fun aProviderThatIsNotAvailableHereCannotFeedAnything() {
        // Registration is a claim; availability is the fact. A pack compiled for a CPU feature this
        // device lacks must not make the resolver keep an encoding nothing can compute (§5.2, #920).
        KernelRegistry.clearForTesting()
        KernelRegistry.register(OneEncodingProvider("Q4_K", available = false))

        assertFalse(RegistryKernelCapabilities.canFeedMatmul(TensorEncoding.Q4_K))
    }

    @Test
    fun aFeedableBlockedEncodingWantsKernelFeedOrder() {
        KernelRegistry.clearForTesting()
        KernelRegistry.register(OneEncodingProvider("Q8_0"))

        assertTrue(
            RegistryKernelCapabilities.wantsKernelFeedOrder(TensorEncoding.Q8_0),
            "every packed matmul kernel in the tree reads input-block-major bytes (#973)",
        )
        assertFalse(
            RegistryKernelCapabilities.wantsKernelFeedOrder(TensorEncoding.Q4_K),
            "an encoding it cannot feed has no order preference to state",
        )
    }

    @Test
    fun denseIsAnsweredByTheFp32Kernel() {
        KernelRegistry.clearForTesting()
        assertFalse(RegistryKernelCapabilities.canFeedMatmul(null), "an empty registry can feed nothing")

        KernelPacks.installReference()
        assertTrue(RegistryKernelCapabilities.canFeedMatmul(null), "the reference pack always carries FP32")
    }
}
