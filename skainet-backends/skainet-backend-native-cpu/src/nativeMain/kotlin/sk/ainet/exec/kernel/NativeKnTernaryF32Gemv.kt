package sk.ainet.exec.kernel

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import sk.ainet.backend.api.kernel.TernaryF32GemvNative
import sk.ainet.backend.api.kernel.TernaryF32KernelPack
import sk.ainet.kernels.cinterop.skainet_ternary_f32_gemv

/**
 * Kotlin/Native face of the vendored NeoGPU LUT kernel (#1139): calls
 * `skainet_ternary_f32_gemv` through cinterop, linked from the static archive
 * `libskainet_kernels.a` — the same C the JVM consumes via FFM and Android via
 * JNI.
 *
 * This is the board-consumption path: a linuxArm64 binary on a Pi-4/Cortex-A72
 * links the archive whose vendored file is pinned to `-march=armv8-a`, so the
 * NEON LUT path runs on dotprod-less cores — the kernel's whole reason to
 * exist. The arrays are pinned and base pointers passed; the C side applies
 * the offsets, no copy is made.
 */
@OptIn(ExperimentalForeignApi::class)
public object NativeKnTernaryF32Gemv : TernaryF32GemvNative {

    override val name: String get() = "cinterop"

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
        require(inputDim % 4 == 0) {
            "NativeKnTernaryF32Gemv: inputDim must be a multiple of 4; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) {
            // The C kernel writes 0.0f per row; mirror that without pinning
            // empty arrays (addressOf(0) needs at least one element).
            out.fill(0f, outOffset, outOffset + outputDim)
            return
        }
        activation.usePinned { inPin ->
            weight.usePinned { wPin ->
                out.usePinned { outPin ->
                    skainet_ternary_f32_gemv(
                        inPin.addressOf(0),
                        activationOffset,
                        wPin.addressOf(0).reinterpret(),
                        weightByteOffset,
                        inputDim,
                        outputDim,
                        outPin.addressOf(0),
                        outOffset,
                    )
                }
            }
        }
    }

    /**
     * Install this kernel into the dispatcher. The archive is linked into the
     * binary, so unlike the FFM/JNI faces there is no missing-artifact case —
     * still routed through [TernaryF32KernelPack.install] so the contract
     * (and the returned serving name) stays uniform across bridges.
     */
    public fun install(warn: (String) -> Unit = {}): String =
        TernaryF32KernelPack.install(this, warn = warn)
}
