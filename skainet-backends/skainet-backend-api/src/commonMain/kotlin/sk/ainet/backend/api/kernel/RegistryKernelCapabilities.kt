package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.plan.KernelCapabilities
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * [KernelCapabilities] answered by the kernels actually registered on this device (#1109).
 *
 * Deliberately thin. `KernelProvider.supports("matmul", [input, weight])` already exists and is
 * already the contract providers override when they ship kernels beyond the built-in accessors —
 * ternary packs among them — so asking it is asking the same source dispatch will ask. Inventing a
 * second capability table beside that one is how the two drift apart.
 *
 * ## Both registries, because there are two
 *
 * A matmul kernel can reach this device by either of two routes, and a capability answer that knows
 * about one of them is wrong about the other. [KernelRegistry] holds the provider SPI — the
 * per-encoding accessors and `supports` — and is what the eager quantized paths consult.
 * [KernelDispatch] holds `KernelKey`-addressed view kernels, and is where `KernelPacks.installReference`
 * puts the reference FP32 GEMM that every target is supposed to have. Ask only the first and a
 * target carrying nothing but the reference pack is reported as unable to multiply dense floats,
 * which is both false and exactly the kind of drift this object exists to avoid.
 *
 * Availability matters as much as registration: a provider whose `isAvailable()` is false on this
 * CPU cannot feed anything, whatever it declares. Dispatch kernels carry their requirements in
 * their key's capability set instead, and are filtered when they are selected.
 */
public object RegistryKernelCapabilities : KernelCapabilities {

    /** The activation dtype every packed matmul kernel in the tree takes. */
    private const val FP32_KEY: String = "Float32"

    override fun canFeedMatmul(encoding: TensorEncoding?): Boolean {
        val weightKey = when (encoding) {
            null -> FP32_KEY
            is TensorEncoding.Dense -> FP32_KEY
            else -> encoding.name
        }
        val fromProviders = KernelRegistry.providers().any { provider ->
            provider.isAvailable() && provider.supports("matmul", listOf(FP32_KEY, weightKey))
        }
        return fromProviders || dispatchHasMatmulFor(weightKey)
    }

    /** Whether a registered [ViewKernel] computes a matmul whose weight operand is [weightKey]. */
    private fun dispatchHasMatmulFor(weightKey: String): Boolean =
        KernelDispatch.kernels().any { kernel ->
            kernel.key.op == "matmul" && kernel.key.operands.any { it.format.kernelEncodingName == weightKey }
        }
}
