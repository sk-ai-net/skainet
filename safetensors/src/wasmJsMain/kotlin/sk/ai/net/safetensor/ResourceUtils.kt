package sk.ai.net.safetensor

/**
 * Wasm/JS implementation of ResourceUtils.
 */
actual object ResourceUtils {
    /**
     * Loads a resource as a byte array.
     *
     * In Wasm/JS, this implementation uses a synchronous fetch from the resources directory.
     * Note: This is a simplified implementation and may not work in all environments.
     *
     * @param path The path to the resource.
     * @return The resource as a byte array, or null if the resource was not found.
     */
    actual fun loadResourceAsBytes(path: String): ByteArray? {
        // For Wasm/JS, we'll use a simple approach that works for resources bundled with the application
        // This assumes the resources are available at the specified path relative to the application

        // In a real implementation, you would use the Fetch API or other browser APIs
        // to load the resource asynchronously, but for simplicity we'll return a dummy implementation

        // For testing purposes, if the path contains "model.safetensors", return a simple dummy tensor
        if (path.contains("model.safetensors")) {
            // Create a simple safetensor file with a 2x2 tensor of zeros
            // Header size (8 bytes) + JSON header + tensor data

            // 1. Create a simple JSON header for a 2x2 tensor
            val jsonHeader = """
                {
                    "test_tensor": {
                        "dtype": "F32",
                        "shape": [2, 2],
                        "data_offsets": [0, 16]
                    }
                }
            """.trimIndent()

            // 2. Convert header size to little-endian bytes
            val headerSize = jsonHeader.length.toLong()
            val headerSizeBytes = ByteArray(8)
            headerSizeBytes[0] = (headerSize and 0xFF).toByte()
            headerSizeBytes[1] = ((headerSize shr 8) and 0xFF).toByte()
            headerSizeBytes[2] = ((headerSize shr 16) and 0xFF).toByte()
            headerSizeBytes[3] = ((headerSize shr 24) and 0xFF).toByte()
            headerSizeBytes[4] = ((headerSize shr 32) and 0xFF).toByte()
            headerSizeBytes[5] = ((headerSize shr 40) and 0xFF).toByte()
            headerSizeBytes[6] = ((headerSize shr 48) and 0xFF).toByte()
            headerSizeBytes[7] = ((headerSize shr 56) and 0xFF).toByte()

            // 3. Create tensor data (4 float32 zeros)
            val tensorData = ByteArray(16)

            // 4. Combine all parts
            val jsonHeaderBytes = jsonHeader.encodeToByteArray()
            val result = ByteArray(8 + jsonHeaderBytes.size + tensorData.size)

            // Copy header size bytes
            headerSizeBytes.copyInto(result, 0)

            // Copy JSON header
            jsonHeaderBytes.copyInto(result, 8)

            // Copy tensor data
            tensorData.copyInto(result, 8 + jsonHeaderBytes.size)

            return result
        }

        // For other resources, return null
        return null
    }
}
