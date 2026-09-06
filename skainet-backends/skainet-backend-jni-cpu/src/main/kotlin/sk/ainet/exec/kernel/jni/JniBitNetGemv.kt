package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.BitNetGemvNative
import sk.ainet.backend.api.kernel.TernaryKernelPacks

/**
 * The NEON `bitnet_gemv` as a [BitNetGemvNative] (SKEEP-003 §5.2/§5.3, #1041, M2-F4).
 *
 * The maths lives in `bitnet_gemv.c`, shared with every other consumer of those kernels; this is
 * the Android/JNI face of it. Whether the loaded library was built with ARMv8.2 dot-product
 * instructions is what [JniKernels.variant] already decided from `/proc/cpuinfo`, so the name — and
 * the capability the kernel is registered with — follows that decision rather than guessing again.
 */
public object JniBitNetGemv : BitNetGemvNative {

    override val name: String
        get() = when (JniKernels.variant) {
            JniKernels.Variant.V82_DOTPROD -> "neon-dotprod"
            JniKernels.Variant.BASELINE -> "neon"
            null -> "unloaded"
        }

    override fun gemvTq2_0(
        activation: ByteArray,
        activationOffset: Int,
        activationScale: Float,
        weight: ByteArray,
        weightByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
        out: FloatArray,
        outOffset: Int,
    ) {
        JniKernels.bitnetGemvTq20(
            activation, activationOffset, activationScale,
            weight, weightByteOffset,
            inputDim, outputDim,
            out, outOffset,
        )
    }

    /**
     * Install this kernel into the dispatcher, or leave the reference in place and say so.
     *
     * Removing the AAR is a supported configuration: dispatch keeps working through the portable
     * kernel, decode gets slower, and [warn] is where that shows up — never a crash (M2-F4).
     *
     * @return the name of the kernel that will serve TQ2_0 matmuls
     */
    public fun install(warn: (String) -> Unit = { println("[skainet] $it") }): String =
        if (JniKernels.isLoaded) {
            TernaryKernelPacks.install(
                native = this,
                capabilities = if (JniKernels.variant == JniKernels.Variant.V82_DOTPROD) {
                    setOf(TernaryKernelPacks.CAPABILITY_DOTPROD)
                } else {
                    emptySet()
                },
                warn = warn,
            )
        } else {
            TernaryKernelPacks.install(native = null, warn = warn)
        }
}
