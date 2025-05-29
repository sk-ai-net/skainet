package sk.ai.net.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import sk.ai.net.graph.tensor.shape.Shape

class SafeTensorReaderTest {
    @Test
    fun testReadTensorFromByteArray() {
        // Create a mock safetensor file
        val mockSafeTensorData = createMockSafeTensorData()

        // Create a SafeTensorReader from the mock data
        val reader = SafeTensorsReader.fromByteArray(mockSafeTensorData)

        // Test getTensorNames
        val tensorNames = reader.getTensorNames()
        assertTrue(tensorNames.isNotEmpty(), "Tensor names should not be empty")

        // Test readTensor
        val tensor = reader.readTensor(tensorNames.first())
        assertNotNull(tensor, "Tensor should not be null")

        // Test tensor shape
        assertEquals(Shape(2, 2), tensor.shape, "Tensor shape should be [2, 2]")
    }

    @Test
    fun testReadSimpleSafeTensors() {
        // Create a mock safetensor file that matches the Python example
        val mockSafeTensorData = createSimpleSafeTensorData()

        // Create a SafeTensorReader from the mock data
        val reader = SafeTensorsReader.fromByteArray(mockSafeTensorData)

        // Test getTensorNames - should have "a" and "b"
        val tensorNames = reader.getTensorNames()
        assertEquals(2, tensorNames.size, "Should have 2 tensors")
        assertTrue(tensorNames.contains("a"), "Should contain tensor 'a'")
        assertTrue(tensorNames.contains("b"), "Should contain tensor 'b'")

        // Test tensor "a" - should be 2x2 zeros of type F32
        val tensorA = reader.readTensor("a")
        assertNotNull(tensorA, "Tensor 'a' should not be null")
        assertEquals(Shape(2, 2), tensorA.shape, "Tensor 'a' shape should be [2, 2]")

        // Check all values are 0.0
        for (i in 0 until 2) {
            for (j in 0 until 2) {
                assertEquals(0.0, tensorA[i, j], "Tensor 'a' value at [$i, $j] should be 0.0")
            }
        }

        // Test tensor "b" - should be 2x3 zeros of type UI8
        val tensorB = reader.readTensor("b")
        assertNotNull(tensorB, "Tensor 'b' should not be null")
        assertEquals(Shape(2, 3), tensorB.shape, "Tensor 'b' shape should be [2, 3]")

        // Check all values are 0.0
        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(0.0, tensorB[i, j], "Tensor 'b' value at [$i, $j] should be 0.0")
            }
        }
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

    /**
     * Creates a mock safetensor file that matches the Python example from the issue description.
     * 
     * The file contains two tensors:
     * - "a": a 2x2 tensor of zeros with dtype F32
     * - "b": a 2x3 tensor of zeros with dtype UI8
     * 
     * This simulates the file created by the Python code:
     * ```python
     * from safetensors.torch import save_file, load_file
     * import torch
     * 
     * tensors = {
     *    "a": torch.zeros((2, 2)),
     *    "b": torch.zeros((2, 3), dtype=torch.uint8)
     * }
     * 
     * save_file(tensors, "./.safetensors")
     * ```
     * 
     * @return A byte array containing the mock safetensor file
     */
    private fun createSimpleSafeTensorData(): ByteArray {
        // Create a JSON header for two tensors: "a" (2x2 zeros, F32) and "b" (2x3 zeros, UI8)
        // The data_offsets indicate where each tensor's data starts and ends in the binary section
        val jsonHeader = """
            {
                "a": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [0, 16]
                },
                "b": {
                    "dtype": "U8",
                    "shape": [2, 3],
                    "data_offsets": [16, 22]
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

        // Create tensor data for "a" (4 float32 zeros)
        val tensorAData = ByteArray(16)
        // All zeros in IEEE 754 binary32 format (0.0f is represented as 0x00000000)
        for (i in 0 until 16) {
            tensorAData[i] = 0x00.toByte()
        }

        // Create tensor data for "b" (6 uint8 zeros)
        val tensorBData = ByteArray(6)
        // All zeros in uint8 format
        for (i in 0 until 6) {
            tensorBData[i] = 0x00.toByte()
        }

        // Combine all parts
        val jsonHeaderBytes = jsonHeader.encodeToByteArray()
        val result = ByteArray(8 + jsonHeaderBytes.size + tensorAData.size + tensorBData.size)

        // Copy header size bytes
        headerSizeBytes.copyInto(result, 0)

        // Copy JSON header
        jsonHeaderBytes.copyInto(result, 8)

        // Copy tensor "a" data
        tensorAData.copyInto(result, 8 + jsonHeaderBytes.size)

        // Copy tensor "b" data
        tensorBData.copyInto(result, 8 + jsonHeaderBytes.size + tensorAData.size)

        return result
    }
}
