package sk.ai.net.safetensors

import kotlinx.cinterop.*
import platform.posix.*

/**
 * Linux implementation of ResourceUtils.
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
        // Open the file
        val file = fopen(path, "rb") ?: return null

        try {
            // Get the file size
            fseek(file, 0, SEEK_END)
            val fileSize = ftell(file)
            rewind(file)

            // Read the file data
            val buffer = ByteArray(fileSize.toInt())
            buffer.usePinned { pinned ->
                val bytesRead = fread(pinned.addressOf(0), 1u, fileSize.toULong(), file)
                if (bytesRead != fileSize.toULong()) {
                    return null
                }
            }

            return buffer
        } finally {
            // Close the file
            fclose(file)
        }
    }
}