package sk.ainet.lang.memory

/** No property store here: debug mode is opt-in through [MemoryDebug.overrideEnabled]. */
internal actual fun platformMemoryDebugEnabled(): Boolean = false

/** No cheap stack walk on this target. */
internal actual fun platformCallSite(): String? = null
