package sk.ai.net.graph.backend

/**
 * Initializes the backend for the Android platform.
 *
 * This class is responsible for initializing the backend with the Android-specific
 * implementations of tensor operations.
 */
object AndroidBackendInitializer {
    /**
     * Initializes the backend.
     *
     * This method is called during application startup to initialize the backend
     * with the Android-specific implementations of tensor operations.
     */
    fun initialize() {
        // Register the Android-specific CPU backend
        val androidCpuBackend = AndroidCpuBackend()
        BackendFactory.registerBackend(BackendFactory.BackendType.CPU, androidCpuBackend)
        
        // Set the default backend type
        BackendFactory.setDefaultBackendType(BackendFactory.BackendType.CPU)
    }
}