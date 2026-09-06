package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.TernaryF32GemvNative
import sk.ainet.backend.api.kernel.TernaryF32KernelPack

/**
 * The vendored NeoGPU LUT kernel as a [TernaryF32GemvNative] (#1139) — the Android/JNI face of
 * `skainet_ternary_f32_gemv`, sharing its C with the FFM and cinterop consumers.
 *
 * Unlike [JniBitNetGemv] there is no capability split: the LUT kernel needs only baseline NEON
 * (architecturally guaranteed on AArch64), so the BASELINE `libskainet_jni.so` — the one every
 * arm64 device can load, Cortex-A72/Pi-class included — carries the full SIMD path. Whichever
 * variant the loader picked, the kernel is the same.
 */
public object JniTernaryF32Gemv : TernaryF32GemvNative {

    override val name: String get() = if (JniKernels.isLoaded) "neon" else "unloaded"

    override fun gemvPacked(
        activation: FloatArray,
        activationOffset: Int,
        weight: ByteArray,
        weightByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
        out: FloatArray,
        outOffset: Int,
    ) {
        JniKernels.ternaryF32Gemv(
            activation, activationOffset,
            weight, weightByteOffset,
            inputDim, outputDim,
            out, outOffset,
        )
    }

    /**
     * Install this kernel into the dispatcher, or leave the int8-requantize path serving and say
     * so. Removing the AAR is a supported configuration — a notice through [warn], never a crash.
     *
     * @return the name of the kernel serving the exact FP32×b1.58 key, or
     *   [TernaryF32KernelPack.NOT_INSTALLED]
     */
    public fun install(warn: (String) -> Unit = { println("[skainet] $it") }): String =
        TernaryF32KernelPack.install(if (JniKernels.isLoaded) this else null, warn = warn)
}
