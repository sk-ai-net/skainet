package sk.ainet.exec.kernel.jni

import sk.ainet.backend.api.kernel.ViewKernelPack

/**
 * `ServiceLoader`-friendly wrapper around [JniMappedKernelPack], the Android counterpart of
 * `FfmRowMajorKernelPackFactory`: it lets `KernelDispatch.ensureInstalled()` discover the JNI
 * row-major kernels so an Android consumer gets zero-copy mapped weights without an explicit
 * bootstrap call.
 *
 * Listed in `META-INF/services/sk.ainet.backend.api.kernel.ViewKernelPack`. Note that Android
 * packaging must preserve `META-INF/services` entries for discovery to work; a consumer whose
 * build strips them can still call [JniMappedKernelPack.install] directly.
 */
public class JniMappedKernelPackFactory : ViewKernelPack {
    override val name: String get() = "jni-rowmajor"
    override fun install(): Unit = JniMappedKernelPack.install()
}
