package sk.ainet.backend.api.kernel

import java.util.ServiceLoader

/**
 * Android discovery for [ViewKernelPack]. `ServiceLoader` exists on Android, so the JNI packs a
 * consumer ships (e.g. the NEON row-major pack in `skainet-backend-jni-cpu`) are discovered the
 * same way as on the JVM, provided the packaging step keeps `META-INF/services` entries.
 */
internal actual fun installPlatformKernelPacks(): List<String> =
    runCatching {
        ServiceLoader.load(ViewKernelPack::class.java)
            .mapNotNull { pack -> runCatching { pack.install(); pack.name }.getOrNull() }
            .toList()
    }.getOrElse { emptyList() }

/**
 * Provider discovery, inlined rather than delegated to `KernelServiceLoader`: that object lives in
 * `jvmMain`, which the Android source set does not see. Same two steps it performs — discover, then
 * register, letting [KernelRegistry] sort by priority on insertion.
 */
internal actual fun installPlatformKernelProviders(): List<String> =
    runCatching {
        ServiceLoader.load(KernelProvider::class.java)
            .mapNotNull { provider -> runCatching { KernelRegistry.register(provider); provider.name }.getOrNull() }
            .toList()
    }.getOrElse { emptyList() }
