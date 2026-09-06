package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.ViewKernelPack

/**
 * `ServiceLoader` entry for the ternary kernels on Android (#1240) — the JNI counterpart of
 * `FfmTernaryKernelPackFactory`: lets `KernelDispatch.ensureInstalled()` reach the exact
 * FP32×`BITNET_B1_58` gemv ([JniTernaryF32Gemv]) and the fused `BITNET_PLANES` lm_head
 * ([JniTernaryLmhead]) without an explicit bootstrap call.
 *
 * Listed in `META-INF/services/sk.ainet.backend.api.kernel.ViewKernelPack`. Both installs
 * register nothing when the bundled `.so` is unavailable; the same packaging caveat as
 * [JniMappedKernelPackFactory] applies — a build that strips `META-INF/services` can still
 * call the installs directly.
 */
public class JniTernaryKernelPackFactory : ViewKernelPack {
    override val name: String get() = "jni-ternary"
    override fun install() {
        JniTernaryF32Gemv.install(warn = {})
        JniTernaryLmhead.install(warn = {})
    }
}
