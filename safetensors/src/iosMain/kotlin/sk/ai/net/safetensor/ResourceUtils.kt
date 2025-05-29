package sk.ai.net.safetensors

import platform.Foundation.*
import kotlinx.cinterop.*

/**
 * iOS implementation of ResourceUtils.
 */
actual object ResourceUtils {
    /**
     * Loads a resource as a byte array.
     *
     * @param path The path to the resource.
     * @return The resource as a byte array, or null if the resource was not found.
     */
    @OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
    actual fun loadResourceAsBytes(path: String): ByteArray? {
        // Split the path to get the resource name and extension
        val components = path.split(".")
        if (components.size < 2) return null

        val resourceName = components.dropLast(1).joinToString(".")
        val resourceType = components.last()

        // Get the path to the resource
        val resourcePath = NSBundle.mainBundle.pathForResource(resourceName, resourceType)
            ?: return null

        // Read the file data
        val fileData = NSData.dataWithContentsOfFile(resourcePath)
            ?: return null

        // Convert NSData to ByteArray
        val length = fileData.length.toInt()
        return ByteArray(length).apply {
            usePinned { pinned ->
                fileData.getBytes(pinned.addressOf(0), length.toULong())
            }
        }
    }
}
