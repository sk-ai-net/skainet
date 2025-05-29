package sk.ai.net.graph.backend

/**
 * Initializes the backend for the WebAssembly platform.
 *
 * This class is responsible for initializing the backend with the WebAssembly-specific
 * implementations of tensor operations.
 */
object WasmBackendInitializer {
    /**
     * Initializes the backend.
     *
     * This method is called during application startup to initialize the backend
     * with the WebAssembly-specific implementations of tensor operations.
     */
    fun initialize() {
        // Register the WebAssembly-specific CPU backend
        val wasmCpuBackend = WasmCpuBackend()
        BackendFactory.registerBackend(BackendFactory.BackendType.CPU, wasmCpuBackend)
        
        // Set the default backend type
        BackendFactory.setDefaultBackendType(BackendFactory.BackendType.CPU)
    }
}