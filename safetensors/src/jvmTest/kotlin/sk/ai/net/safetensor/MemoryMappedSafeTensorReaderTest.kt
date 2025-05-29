package sk.ai.net.safetensors

import java.io.File
import java.nio.ByteOrder
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ai.net.graph.tensor.shape.Shape

class MemoryMappedSafeTensorReaderTest {
    @Test
    fun testMemoryMappedLoading() {
        // Create a temporary file with mock safetensor data
        val tempFile = createTempSafeTensorFile()

        try {
            // Create a MemoryMappedSafeTensorReader from the file
            val reader = fromFile(tempFile)

            // Test getTensorNames
            val tensorNames = reader.getTensorNames()
            assertTrue(tensorNames.isNotEmpty(), "Tensor names should not be empty")

            // Test readTensor
            val tensor = reader.readTensor(tensorNames.first())
            assertNotNull(tensor, "Tensor should not be null")

            // Test tensor shape
            assertEquals(Shape(2, 2), tensor.shape, "Tensor shape should be [2, 2]")

            // Test tensor values
            assertEquals(1.0, tensor[0, 0], 0.001, "Tensor value at [0, 0] should be 1.0")
            assertEquals(2.0, tensor[0, 1], 0.001, "Tensor value at [0, 1] should be 2.0")
            assertEquals(3.0, tensor[1, 0], 0.001, "Tensor value at [1, 0] should be 3.0")
            assertEquals(4.0, tensor[1, 1], 0.001, "Tensor value at [1, 1] should be 4.0")
        } finally {
            // Clean up the temporary file
            tempFile.delete()
        }
    }

    @Test
    fun testMemoryMappedLoadingFromPath() {
        // Create a temporary file with mock safetensor data
        val tempFile = createTempSafeTensorFile()

        try {
            // Create a MemoryMappedSafeTensorReader from the file path
            val reader = fromFilePath(tempFile.absolutePath)

            // Test getTensorNames
            val tensorNames = reader.getTensorNames()
            assertTrue(tensorNames.isNotEmpty(), "Tensor names should not be empty")

            // Test readTensor
            val tensor = reader.readTensor(tensorNames.first())
            assertNotNull(tensor, "Tensor should not be null")

            // Test tensor shape
            assertEquals(Shape(2, 2), tensor.shape, "Tensor shape should be [2, 2]")
        } finally {
            // Clean up the temporary file
            tempFile.delete()
        }
    }

    /**
     * Creates a temporary file with mock safetensor data.
     * 
     * @return The temporary file.
     */
    private fun createTempSafeTensorFile(): File {
        // Create a temporary file
        val tempFile = Files.createTempFile("safetensor", ".safetensors").toFile()

        // Create mock safetensor data
        val mockData = createMockSafeTensorData()

        // Write the data to the file
        tempFile.writeBytes(mockData)

        return tempFile
    }

    /**
     * Creates mock safetensor data with a single 2x2 tensor.
     * 
     * @return The mock safetensor data as a byte array.
     */
    private fun createMockSafeTensorData(): ByteArray {
        // Create a simple JSON header for a 2x2 tensor
        val jsonHeader = """
            {
                "test_tensor": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [0, 16]
                }
            }
        """.trimIndent()

        // Convert header size to little-endian bytes
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

        // Create tensor data (4 float32 values: 1.0, 2.0, 3.0, 4.0)
        val tensorData = ByteArray(16)
        // 1.0f in IEEE 754 binary32 format
        tensorData[0] = 0x00.toByte()
        tensorData[1] = 0x00.toByte()
        tensorData[2] = 0x80.toByte()
        tensorData[3] = 0x3F.toByte()
        // 2.0f in IEEE 754 binary32 format
        tensorData[4] = 0x00.toByte()
        tensorData[5] = 0x00.toByte()
        tensorData[6] = 0x00.toByte()
        tensorData[7] = 0x40.toByte()
        // 3.0f in IEEE 754 binary32 format
        tensorData[8] = 0x00.toByte()
        tensorData[9] = 0x00.toByte()
        tensorData[10] = 0x40.toByte()
        tensorData[11] = 0x40.toByte()
        // 4.0f in IEEE 754 binary32 format
        tensorData[12] = 0x00.toByte()
        tensorData[13] = 0x00.toByte()
        tensorData[14] = 0x80.toByte()
        tensorData[15] = 0x40.toByte()

        // Combine all parts
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
}
