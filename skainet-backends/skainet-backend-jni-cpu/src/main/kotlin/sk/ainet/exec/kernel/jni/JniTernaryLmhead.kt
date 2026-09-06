package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.TernaryLmheadNative
import sk.ainet.backend.api.kernel.TernaryPlanesKernelPack

/**
 * The vendored NeoGPU fused 4-plane lm_head as a [TernaryLmheadNative] (#1150) — the Android/JNI
 * face of `skainet_ternary_lmhead_stage1`, sharing its C with the FFM and cinterop consumers.
 *
 * Like [JniTernaryF32Gemv], no capability split: the LUT kernel needs only baseline NEON, so the
 * BASELINE `libskainet_jni.so` carries the full SIMD path on every arm64 device.
 */
public object JniTernaryLmhead : TernaryLmheadNative {

    override val name: String get() = if (JniKernels.isLoaded) "neon" else "unloaded"

    override fun lmheadStage1(
        activation: FloatArray, activationOffset: Int,
        weight: ByteArray, planesByteOffset: Int,
        planeStrideBytes: Int, rowScaleByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        JniKernels.ternaryLmheadStage1(
            activation, activationOffset,
            weight, planesByteOffset, planeStrideBytes, rowScaleByteOffset,
            inputDim, outputDim,
            out, outOffset,
        )
    }

    /**
     * Install this kernel into the dispatcher, or leave the decoding reference serving and say
     * so — removing the AAR is a supported configuration, never a crash.
     */
    public fun install(warn: (String) -> Unit = { println("[skainet] $it") }): String =
        TernaryPlanesKernelPack.install(if (JniKernels.isLoaded) this else null, warn = warn)
}
