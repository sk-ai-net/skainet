package sk.ainet.backend.api.kernel


internal actual fun platformUseRegistry(): Boolean = System.getProperty(DispatchMode.PROPERTY) != "false"
