package sk.ainet.backend.api.kernel

import java.util.ServiceLoader

/**
 * JVM discovery for [ViewKernelPack], mirroring [KernelServiceLoader]'s handling of
 * [KernelProvider]: a backend module declares its pack in
 * `META-INF/services/sk.ainet.backend.api.kernel.ViewKernelPack` and it is installed automatically
 * the first time [KernelDispatch] needs kernels.
 *
 * A pack that throws while installing is skipped rather than allowed to break dispatch — a broken
 * optional backend must not take the process down, and the reference kernel still serves every
 * format correctly.
 */
internal actual fun installPlatformKernelPacks(): List<String> =
    runCatching {
        ServiceLoader.load(ViewKernelPack::class.java)
            .mapNotNull { pack -> runCatching { pack.install(); pack.name }.getOrNull() }
            .toList()
    }.getOrElse { emptyList() }

internal actual fun installPlatformKernelProviders(): List<String> =
    runCatching { KernelServiceLoader.installAll() }.getOrElse { emptyList() }
