package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.ViewKernelPack

/**
 * `ServiceLoader`-friendly wrapper around [FfmRowMajorKernelPack] — the same shape
 * [NativeKernelProviderFactory] gives [NativeKernelProvider], because `ServiceLoader` needs a
 * public no-arg constructor and a Kotlin `object` does not expose one.
 *
 * Listed in `META-INF/services/sk.ainet.backend.api.kernel.ViewKernelPack` so
 * `KernelDispatch.ensureInstalled()` finds the FFM row-major kernels without the consumer calling
 * anything. [FfmRowMajorKernelPack.install] is already a no-op when the native library is missing,
 * so discovery on a machine without it costs a lookup and registers nothing.
 */
public class FfmRowMajorKernelPackFactory : ViewKernelPack {
    override val name: String get() = "ffm-rowmajor"
    override fun install(): Unit = FfmRowMajorKernelPack.install()
}
