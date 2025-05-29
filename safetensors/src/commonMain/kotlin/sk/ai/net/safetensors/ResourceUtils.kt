package sk.ai.net.safetensors

/**
 * Utility class for loading resources in a platform-independent way.
 */
expect object ResourceUtils {
    /**
     * Loads a resource as a byte array.
     *
     * @param path The path to the resource.
     * @return The resource as a byte array, or null if the resource was not found.
     */
    fun loadResourceAsBytes(path: String): ByteArray?
}