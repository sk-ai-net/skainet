package sk.ainet.backend.api.kernel


/**
 * Whether the generic (non-fast-path) matmul goes through [KernelDispatch] or the legacy
 * per-element fallback (SKEEP-003 §5.1, migration slice #1028).
 *
 * The registry path is the default: it normalises rank once as views and reads packed operands
 * through decoding `get()`, so a rank-1 decode step against a packed weight is *correct by
 * construction* instead of a `ClassCastException` (#993). The legacy path stays one flag away
 * while the migration settles; it is deleted once the golden parity and benchmark evidence is in.
 */
public object DispatchMode {
    /** Set to `false` (`skainet.dispatch.registry=false`) to force the legacy generic fallback. */
    public const val PROPERTY: String = "skainet.dispatch.registry"

    /** Overridable in tests; `null` means "read the platform setting". */
    public var overrideEnabled: Boolean? = null

    /** Whether the generic path should use the kernel registry. */
    public fun useRegistry(): Boolean = overrideEnabled ?: platformUseRegistry()
}

/** Platform reading of [DispatchMode.PROPERTY]; defaults to `true` where there is no property store. */
internal expect fun platformUseRegistry(): Boolean
