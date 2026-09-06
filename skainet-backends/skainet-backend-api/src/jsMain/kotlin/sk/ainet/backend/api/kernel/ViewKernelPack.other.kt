package sk.ainet.backend.api.kernel


/**
 * No `ServiceLoader` on this platform, so there is nothing to discover: packs are installed
 * explicitly by the consumer (the same split [KernelProvider] documents — see
 * `NativeKnKernelProvider`, which is registered by hand on Kotlin/Native).
 */
internal actual fun installPlatformKernelPacks(): List<String> = emptyList()

internal actual fun installPlatformKernelProviders(): List<String> = emptyList()
