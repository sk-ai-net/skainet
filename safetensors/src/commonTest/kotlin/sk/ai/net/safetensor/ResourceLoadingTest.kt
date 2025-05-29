package sk.ai.net.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ai.net.graph.tensor.shape.Shape

class ResourceLoadingTest {
    @Test
    fun testLoadModelFromByteArray() {
        // Create a mock safetensor file
        val mockSafeTensorData = createMockSafeTensorData()

        // Create a SafeTensorReader from the mock data
        val reader = SafeTensorsReader.fromByteArray(mockSafeTensorData)

        // Verify that the reader was created successfully
        assertNotNull(reader, "SafeTensorReader should be created successfully")

        // Get the tensor names
        val tensorNames = reader.getTensorNames()

        // Verify that there is at least one tensor
        assertTrue(tensorNames.isNotEmpty(), "There should be at least one tensor in the model")

        // Read the first tensor
        val tensor = reader.readTensor(tensorNames.first())

        // Verify that the tensor was read successfully
        assertNotNull(tensor, "Tensor should be read successfully")

        // Verify the tensor shape
        assertEquals(Shape(2, 2), tensor.shape, "Tensor shape should be [2, 2]")

        // Print tensor information for debugging
        println("Tensor name: ${tensorNames.first()}")
        println("Tensor shape: ${tensor.shape}")
    }

    /**
     * Creates a mock safetensor file with a single 2x2 tensor.
     * 
     * The format is:
     * - 8 bytes: header size (little-endian)
     * - JSON header
     * - Tensor data
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
