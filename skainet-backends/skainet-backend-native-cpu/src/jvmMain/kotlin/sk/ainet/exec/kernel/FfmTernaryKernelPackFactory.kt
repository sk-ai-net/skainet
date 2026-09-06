package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.ViewKernelPack

/**
 * `ServiceLoader` entry for the ternary kernels (#1240) — the missing sibling of
 * [FfmRowMajorKernelPackFactory]: `KernelDispatch.ensureInstalled()` discovered the Q-series
 * row-major pack but not the ternary ones, so a consumer loading `BITNET_B1_58` /
 * `BITNET_PLANES` weights silently fell to the int8-requantize or decoding-reference path
 * (~120× slower per the #1141 bench) unless it called the two installs below explicitly —
 * exactly the failure mode the self-healing dispatch exists to eliminate.
 *
 * Installs the exact FP32×`BITNET_B1_58` LUT gemv ([NativeTernaryF32GemvKernel]) and the fused
 * `BITNET_PLANES` lm_head ([NativeTernaryLmheadKernel]). Both delegate to their packs'
 * `install(native?)`, which registers nothing when the bundled native library is missing —
 * discovery on such a machine costs a lookup and leaves dispatch to the reference kernels.
 */
public class FfmTernaryKernelPackFactory : ViewKernelPack {
    override val name: String get() = "ffm-ternary"
    override fun install() {
        NativeTernaryF32GemvKernel.install()
        NativeTernaryLmheadKernel.install()
    }
}
