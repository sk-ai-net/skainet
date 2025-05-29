package sk.ai.net.safetensor

import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.system.measureTimeMillis

/**
 * Tests to demonstrate the memory efficiency of the memory-mapped implementation.
 *
 * These tests create a large safetensor file and compare the memory usage and
 * loading time of the standard implementation vs. the memory-mapped implementation.
 *
 * Note: These tests are more for demonstration purposes and may not be suitable
 * for automated testing, as they depend on system memory and garbage collection.
 */
class MemoryEfficiencyTest {

    /**
     * Test to compare the memory usage and loading time of the standard implementation
     * vs. the memory-mapped implementation.
     *
     * This test creates a large safetensor file (100MB) and loads it using both
     * implementations. It then measures the memory usage and loading time of each.
     *
     * The memory-mapped implementation should use significantly less memory and
     * load faster than the standard implementation.
     */
    @Test
    fun testMemoryEfficiency() {
        // Create a large safetensor file (100MB)
        val tempFile = createLargeSafeTensorFile(100 * 1024 * 1024) // 100MB

        try {
            println("Testing memory efficiency with a 100MB safetensor file")

            // Measure memory usage and loading time for standard implementation
            System.gc() // Try to clean up memory before the test
            val standardMemoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

            val standardLoadTime = measureTimeMillis {
                // Load the file using the standard implementation
                val fileBytes = tempFile.readBytes()
                val standardReader = SafeTensorReader.fromByteArray(fileBytes)

                // Access some tensors to ensure they're loaded
                val tensorNames = standardReader.getTensorNames()
                standardReader.readTensor(tensorNames.first())
            }

            System.gc() // Try to clean up memory after the test
            val standardMemoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val standardMemoryUsage = standardMemoryAfter - standardMemoryBefore

            println("Standard implementation:")
            println("  Loading time: $standardLoadTime ms")
            println("  Memory usage: ${standardMemoryUsage / (1024 * 1024)} MB")

            // Measure memory usage and loading time for memory-mapped implementation
            System.gc() // Try to clean up memory before the test
            val mappedMemoryBefore = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()

            val mappedLoadTime = measureTimeMillis {
                // Load the file using the memory-mapped implementation
                val mappedReader = fromFile(tempFile)

                // Access some tensors to ensure they're loaded
                val tensorNames = mappedReader.getTensorNames()
                mappedReader.readTensor(tensorNames.first())
            }

            System.gc() // Try to clean up memory after the test
            val mappedMemoryAfter = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()
            val mappedMemoryUsage = mappedMemoryAfter - mappedMemoryBefore

            println("Memory-mapped implementation:")
            println("  Loading time: $mappedLoadTime ms")
            println("  Memory usage: ${mappedMemoryUsage / (1024 * 1024)} MB")

            // The memory-mapped implementation should use less memory
            println("Memory usage ratio (mapped/standard): ${mappedMemoryUsage.toDouble() / standardMemoryUsage.toDouble()}")

            // This assertion might not always pass due to JVM memory management,
            // but the memory-mapped implementation should generally use less memory
            assertTrue(mappedMemoryUsage < standardMemoryUsage * 0.9, 
                "Memory-mapped implementation should use less memory than standard implementation")

        } finally {
            // Clean up the temporary file
            tempFile.delete()
        }
    }

    /**
     * Creates a large safetensor file with a single large tensor.
     *
     * @param size The approximate size of the file in bytes.
     * @return The temporary file.
     */
    private fun createLargeSafeTensorFile(size: Int): File {
        // Create a temporary file
        val tempFile = Files.createTempFile("large_safetensor", ".safetensors").toFile()

        // Calculate the size of the tensor data (size - header)
        val headerSize = 100 // Approximate size of the header
        val tensorDataSize = size - headerSize - 8 // 8 bytes for the header size

        // Create a simple JSON header for a large tensor
        val jsonHeader = """
            {
                "large_tensor": {
                    "dtype": "F32",
                    "shape": [${tensorDataSize / 4}, 1],
                    "data_offsets": [0, $tensorDataSize]
                }
            }
        """.trimIndent()

        // Convert header size to little-endian bytes
        val actualHeaderSize = jsonHeader.length.toLong()
        val headerSizeBytes = ByteArray(8)
        headerSizeBytes[0] = (actualHeaderSize and 0xFF).toByte()
        headerSizeBytes[1] = ((actualHeaderSize shr 8) and 0xFF).toByte()
        headerSizeBytes[2] = ((actualHeaderSize shr 16) and 0xFF).toByte()
        headerSizeBytes[3] = ((actualHeaderSize shr 24) and 0xFF).toByte()
        headerSizeBytes[4] = ((actualHeaderSize shr 32) and 0xFF).toByte()
        headerSizeBytes[5] = ((actualHeaderSize shr 40) and 0xFF).toByte()
        headerSizeBytes[6] = ((actualHeaderSize shr 48) and 0xFF).toByte()
        headerSizeBytes[7] = ((actualHeaderSize shr 56) and 0xFF).toByte()

        // Create tensor data (all zeros for simplicity)
        val tensorData = ByteArray(tensorDataSize)

        // Combine all parts
        val jsonHeaderBytes = jsonHeader.encodeToByteArray()
        val result = ByteArray(8 + jsonHeaderBytes.size + tensorData.size)

        // Copy header size bytes
        headerSizeBytes.copyInto(result, 0)

        // Copy JSON header
        jsonHeaderBytes.copyInto(result, 8)

        // Copy tensor data
        tensorData.copyInto(result, 8 + jsonHeaderBytes.size)

        // Write the data to the file
        tempFile.writeBytes(result)

        return tempFile
    }
}
