package sk.ainet.lang.memory

internal actual fun platformMemoryDebugEnabled(): Boolean =
    System.getProperty(MemoryDebug.PROPERTY) == "true" || System.getenv(MemoryDebug.ENV).let { it == "1" || it == "true" }

/** The first frame outside `sk.ainet.lang.memory` — where the allocation was actually asked for. */
internal actual fun platformCallSite(): String? = StackWalker.getInstance().walk { frames ->
    frames.filter { !it.className.startsWith("sk.ainet.lang.memory") }
        .findFirst()
        .map { "${it.className}.${it.methodName}(${it.fileName}:${it.lineNumber})" }
        .orElse(null)
}
