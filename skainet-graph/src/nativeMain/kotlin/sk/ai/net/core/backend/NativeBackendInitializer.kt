package sk.ai.net.core.backend

import sk.ai.net.core.backend.BackendFactory

/**
 * Initializes the backend for the Native platform.
 *
 * This class is responsible for initializing the backend with the Native-specific
 * implementations of tensor operations.
 */
object NativeBackendInitializer {
    /**
     * Initializes the backend.
     *
     * This method is called during application startup to initialize the backend
     * with the Native-specific implementations of tensor operations.
     */
    fun initialize() {
        // Register the Native-specific CPU backend
        val nativeCpuBackend = NativeCpuBackend()
        BackendFactory.registerBackend(BackendFactory.BackendType.CPU, nativeCpuBackend)
        
        // Set the default backend type
        BackendFactory.setDefaultBackendType(BackendFactory.BackendType.CPU)
    }
}