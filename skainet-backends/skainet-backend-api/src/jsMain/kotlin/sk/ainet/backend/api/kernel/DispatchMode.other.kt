package sk.ainet.backend.api.kernel


/** No system properties here: the registry path is always on (override in tests via [DispatchMode.overrideEnabled]). */
internal actual fun platformUseRegistry(): Boolean = true
