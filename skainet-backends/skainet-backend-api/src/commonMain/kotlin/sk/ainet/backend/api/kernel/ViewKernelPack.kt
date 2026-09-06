package sk.ainet.backend.api.kernel


/**
 * A installable set of [ViewKernel]s for [KernelDispatch] — the view-keyed sibling of
 * [KernelProvider], which serves [KernelRegistry].
 *
 * Why this exists: [KernelRegistry] self-heals (an ops instance that finds it empty calls
 * `KernelServiceLoader.installAll()`), but [KernelDispatch] historically did not. Every consumer
 * had to remember two explicit `install()` calls before the first forward pass, and forgetting
 * them is invisible: the dispatcher simply serves the decoding reference kernel, which is correct
 * and roughly a thousand times slower. Making packs *discoverable* lets [KernelDispatch] populate
 * itself the same way the provider registry already does.
 *
 * Implementations must be cheap to construct and idempotent to [install] — the dispatcher may call
 * it once per process, and a pack whose native library or platform feature is unavailable should
 * register nothing rather than throw.
 *
 * **Discovery is JVM-only**, exactly as it is for [KernelProvider]: `ServiceLoader` has no
 * equivalent on Kotlin/Native, wasm or JS, so those platforms install their packs manually (see
 * `installPlatformKernelPacks`).
 */
public interface ViewKernelPack {

    /** Stable identifier, used for logging and de-duplication (e.g. `"ffm-rowmajor"`). */
    public val name: String

    /**
     * Register this pack's kernels into [KernelDispatch]. Must be safe to call more than once and
     * must degrade to a no-op when the platform cannot serve it (missing native library, absent
     * vector unit, …).
     */
    public fun install()
}

/**
 * Discover and install every [ViewKernelPack] this platform exposes, returning the names of the
 * packs that were installed.
 *
 * JVM: `ServiceLoader`-discovered, mirroring [KernelServiceLoader]. Everywhere else: no discovery
 * mechanism exists, so this returns an empty list and the consumer installs packs explicitly.
 */
internal expect fun installPlatformKernelPacks(): List<String>

/**
 * Populate [KernelRegistry] from platform-discovered [KernelProvider]s when it is still empty.
 *
 * Needed because [KernelPacks.install] derives its kernels from `KernelRegistry.bestAvailable()`,
 * which is `null` on an empty registry — a bootstrap that runs before any ops instance exists
 * would otherwise install nothing but the reference kernel.
 */
internal expect fun installPlatformKernelProviders(): List<String>
