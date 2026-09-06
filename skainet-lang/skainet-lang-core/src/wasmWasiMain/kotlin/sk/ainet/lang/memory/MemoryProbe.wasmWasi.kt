package sk.ainet.lang.memory

/** No process-level view here: a browser or Wasm host does not expose one. */
public actual object MemoryProbe {
    public actual fun rssBytes(): Long? = null
    public actual fun majorFaults(): Long? = null
    public actual fun minorFaults(): Long? = null
}
