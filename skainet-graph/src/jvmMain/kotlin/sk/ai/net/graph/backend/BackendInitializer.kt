package sk.ai.net.graph.backend

/**
 * Initializes the backend for the desktop platform.
 *
 * This class is responsible for initializing the backend with the JVM-specific
 * implementations of tensor operations.
 */
object BackendInitializer {
    /**
     * Initializes the backend.
     *
     * This method is called during application startup to initialize the backend
     * with the JVM-specific implementations of tensor operations.
     */
    fun initialize() {
        // Register the JVM-specific CPU backend
        val jvmCpuBackend = JvmCpuBackend()
        BackendFactory.registerBackend(BackendFactory.BackendType.CPU, jvmCpuBackend)
        
        // Set the default backend type
        BackendFactory.setDefaultBackendType(BackendFactory.BackendType.CPU)
    }
}