package sk.ai.net.graph.backend

/**
 * Factory class for creating and managing computation backends.
 *
 * This class provides methods for creating and accessing different computation backends,
 * such as CPU and GPU backends. It also manages a default backend that is used when
 * no specific backend is requested.
 */
object BackendFactory {
    /**
     * The available backend types.
     */
    enum class BackendType {
        CPU,
        GPU
    }

    /**
     * The default backend type.
     */
    private var defaultBackendType = BackendType.CPU

    /**
     * Cache of backend instances.
     */
    private val backends = mutableMapOf<BackendType, ComputeBackend>()

    /**
     * Gets a backend of the specified type.
     *
     * @param type The type of backend to get.
     * @return The backend instance.
     */
    fun getBackend(type: BackendType): ComputeBackend {
        return backends.getOrPut(type) {
            when (type) {
                BackendType.CPU -> CpuBackend()
                BackendType.GPU -> GpuBackend()
            }
        }
    }

    /**
     * Gets the default backend.
     *
     * @return The default backend instance.
     */
    fun getDefaultBackend(): ComputeBackend {
        return getBackend(defaultBackendType)
    }

    /**
     * Sets the default backend type.
     *
     * @param type The new default backend type.
     */
    fun setDefaultBackendType(type: BackendType) {
        defaultBackendType = type
    }

    /**
     * Gets the default backend type.
     *
     * @return The default backend type.
     */
    fun getDefaultBackendType(): BackendType {
        return defaultBackendType
    }

    /**
     * Registers a backend of the specified type.
     *
     * @param type The type of backend to register.
     * @param backend The backend instance to register.
     */
    fun registerBackend(type: BackendType, backend: ComputeBackend) {
        backends[type] = backend
    }
}
