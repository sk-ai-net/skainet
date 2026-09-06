package sk.ainet.exec.kernel

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import sk.ainet.backend.api.kernel.TernaryLmheadNative
import sk.ainet.backend.api.kernel.TernaryPlanesKernelPack
import sk.ainet.kernels.cinterop.skainet_ternary_lmhead_stage1

/**
 * Kotlin/Native face of the vendored NeoGPU fused 4-plane lm_head (#1150): calls
 * `skainet_ternary_lmhead_stage1` through cinterop from the static archive — the linuxArm64
 * (Pi-4/Cortex-A72) consumption path, where the archive's vendored file is pinned to
 * `-march=armv8-a`. The FP16 row scales live inside the pinned weight array; the pointer is
 * derived at `rowScaleByteOffset` (2-byte aligned by the seam's contract — unaligned uint16
 * reads are legal on AArch64 anyway).
 */
@OptIn(ExperimentalForeignApi::class)
public object NativeKnTernaryLmhead : TernaryLmheadNative {

    override val name: String get() = "cinterop"

    override fun lmheadStage1(
        activation: FloatArray, activationOffset: Int,
        weight: ByteArray, planesByteOffset: Int,
        planeStrideBytes: Int, rowScaleByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        require(inputDim % 4 == 0) {
            "NativeKnTernaryLmhead: inputDim must be a multiple of 4; got $inputDim"
        }
        if (outputDim == 0) return
        if (inputDim == 0) {
            // The C kernel writes rowScale * 0 per row; mirror without pinning empty arrays.
            out.fill(0f, outOffset, outOffset + outputDim)
            return
        }
        activation.usePinned { inPin ->
            weight.usePinned { wPin ->
                out.usePinned { outPin ->
                    skainet_ternary_lmhead_stage1(
                        inPin.addressOf(0),
                        activationOffset,
                        wPin.addressOf(0).reinterpret(),
                        planesByteOffset,
                        planeStrideBytes,
                        wPin.addressOf(rowScaleByteOffset).reinterpret(),
                        0,
                        inputDim,
                        outputDim,
                        outPin.addressOf(0),
                        outOffset,
                    )
                }
            }
        }
    }

    /** Install this kernel into the dispatcher — the archive is linked in, so always available. */
    public fun install(warn: (String) -> Unit = {}): String =
        TernaryPlanesKernelPack.install(this, warn = warn)
}
