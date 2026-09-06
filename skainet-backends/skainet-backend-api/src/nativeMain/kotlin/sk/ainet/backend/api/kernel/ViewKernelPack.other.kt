package sk.ainet.backend.api.kernel


/**
 * No `ServiceLoader` on this platform, so there is nothing to discover: packs are installed
 * explicitly by the consumer (the same split [KernelProvider] documents — see
 * `NativeKnKernelProvider`, which is registered by hand on Kotlin/Native). That includes the
 * ternary packs (#1240): a Kotlin/Native consumer of `BITNET_B1_58` / `BITNET_PLANES` weights
 * calls `NativeKnTernaryF32Gemv.install()` / `NativeKnTernaryLmhead.install()` itself, or
 * dispatch serves the decoding reference kernels.
 */
internal actual fun installPlatformKernelPacks(): List<String> = emptyList()

internal actual fun installPlatformKernelProviders(): List<String> = emptyList()
