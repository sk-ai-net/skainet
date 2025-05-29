package sk.ai.net.safetensor

import java.io.InputStream

/**
 * Desktop implementation of ResourceUtils.
 */
actual object ResourceUtils {
    /**
     * Loads a resource as a byte array.
     *
     * @param path The path to the resource.
     * @return The resource as a byte array, or null if the resource was not found.
     */
    actual fun loadResourceAsBytes(path: String): ByteArray? {
        val resourceStream: InputStream? = ResourceUtils::class.java.classLoader.getResourceAsStream(path)
        return resourceStream?.readBytes()
    }
}