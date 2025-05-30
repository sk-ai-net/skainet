package sk.ai.net.safetensors

import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import sk.ai.net.core.tensor.Int8Tensor
import sk.ai.net.core.tensor.SimpleTensor
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.shape.Shape

/**
 * Implementation of SafeTensorsReader that reads from a Source in a streaming manner.
 *
 * This implementation follows the approach shown in the JavaScript code:
 * 1. Read only the first 8 bytes to get the header size
 * 2. Read only enough data to get the complete header
 * 3. Parse the header JSON
 * 4. Read tensor data only when needed
 *
 * This is more efficient for large files than loading the entire file into memory.
 *
 * @property sourceProvider A function that returns a fresh Source each time it's called.
 */
class StreamingSafeTensorsReader(private val sourceProvider: () -> Source) : SafeTensorsReader {
    // Header information
    private val headerSize: Long
    private val headerJson: String
    private val metadata: Map<String, StreamingTensorMetadata>

    init {
        // Get a fresh source
        val source = sourceProvider()

        // Read the header size (first 8 bytes)
        val headerSizeBytes = source.readByteArray(8)
        headerSize = readHeaderSize(headerSizeBytes)

        // Read the header JSON
        val headerBytes = source.readByteArray(headerSize.toInt())
        headerJson = headerBytes.decodeToString()

        // Parse the header JSON
        metadata = parseHeaderJson(headerJson)

        // Close the source
        source.close()
    }

    /**
     * Reads a tensor from the safetensor data.
     *
     * @param name The name of the tensor to read.
     * @return The tensor if found, or null if not found.
     */
    override fun readTensor(name: String): Tensor? {
        val meta = metadata[name] ?: return null

        try {
            // Get a fresh source
            val source = sourceProvider()

            try {
                // Calculate the absolute offset in the file (header size + JSON header + data offset)
                val absoluteOffset = 8 + headerSize + meta.dataOffset
                val dataLength = meta.dataLength.toInt()

                // Validate the offset and length
                if (absoluteOffset < 0) {
                    println("[DEBUG] Invalid tensor offset: $absoluteOffset")
                    throw IllegalArgumentException("Invalid tensor offset: $absoluteOffset")
                }

                // Skip to the tensor data
                source.skip(absoluteOffset)

                // Read the tensor data
                val tensorData = source.readByteArray(dataLength)

                // Create a tensor based on the dtype and shape
                return when (meta.dtype) {
                    "F32" -> {
                        // Validate that the tensor data size is a multiple of 4 (size of float)
                        if (tensorData.size % 4 != 0) {
                            println("[DEBUG] Invalid F32 tensor data size: ${tensorData.size} (not a multiple of 4)")
                            throw IllegalArgumentException("Invalid F32 tensor data size: ${tensorData.size}")
                        }

                        // Convert byte array to double array for F32 data
                        val doubleData = DoubleArray(tensorData.size / 4)
                        for (i in doubleData.indices) {
                            val startIndex = i * 4
                            if (startIndex + 4 > tensorData.size) {
                                println("[DEBUG] Index out of bounds: $startIndex + 4 > ${tensorData.size}")
                                throw IndexOutOfBoundsException("Index out of bounds: $startIndex + 4 > ${tensorData.size}")
                            }

                            val bytes = tensorData.sliceArray(startIndex until (startIndex + 4))
                            val floatBits = (bytes[0].toInt() and 0xFF) or
                                            ((bytes[1].toInt() and 0xFF) shl 8) or
                                            ((bytes[2].toInt() and 0xFF) shl 16) or
                                            ((bytes[3].toInt() and 0xFF) shl 24)
                            doubleData[i] = Float.fromBits(floatBits).toDouble()
                        }
                        SimpleTensor(meta.shape, doubleData)
                    }
                    "I8" -> {
                        // For I8 data, we can use Int8Tensor with appropriate scale and zero point
                        // For simplicity, we're using a default scale and zero point here
                        // In a real implementation, these would be extracted from the metadata
                        val scale = 1.0
                        val zeroPoint = 0
                        Int8Tensor(meta.shape, tensorData, scale, zeroPoint)
                    }
                    else -> {
                        println("[DEBUG] Unsupported tensor dtype: ${meta.dtype}")
                        // For unsupported dtypes, convert to double array with zeros
                        val totalElements = meta.shape.volume
                        SimpleTensor(meta.shape, DoubleArray(totalElements))
                    }
                }
            } finally {
                // Always close the source
                source.close()
            }
        } catch (e: Exception) {
            println("[DEBUG] Error reading tensor data: ${e.message}")
            throw e
        }
    }

    /**
     * Gets all tensor names available in the safetensor file.
     *
     * @return A list of tensor names.
     */
    override fun getTensorNames(): List<String> {
        try {
            return metadata.keys.toList()
        } catch (e: Exception) {
            println("[DEBUG] Error getting tensor names: ${e.message}")
            return emptyList()
        }
    }

    /**
     * Gets the metadata from the safetensor file.
     *
     * @return A map of metadata keys to values.
     */
    override fun getMetadata(): Map<String, String> {
        // SafeTensors files don't have a standard metadata format like GGUF files
        return emptyMap()
    }

    /**
     * Gets the JSON header from the safetensor file.
     *
     * @return The JSON header string.
     */
    override fun getHeaderJson(): String {
        return headerJson
    }

    /**
     * Reads the header size from the first 8 bytes of the safetensor file.
     * 
     * The header size is stored as a 64-bit little-endian unsigned integer
     * at the beginning of the file.
     * 
     * @param data The byte array containing the first 8 bytes of the safetensor data.
     * @return The size of the JSON header in bytes.
     */
    private fun readHeaderSize(data: ByteArray): Long {
        // Check if we have enough data to read the header size
        if (data.size < 8) {
            println("[DEBUG] Not enough data to read header size: ${data.size} bytes")
            return 0
        }

        // Print the first 8 bytes for debugging
        println("[DEBUG] First 8 bytes: ${data[0]}, ${data[1]}, ${data[2]}, ${data[3]}, ${data[4]}, ${data[5]}, ${data[6]}, ${data[7]}")

        // The first 8 bytes contain the header size as a 64-bit little-endian unsigned integer
        val headerSize = (data[0].toLong() and 0xFF) or
               ((data[1].toLong() and 0xFF) shl 8) or
               ((data[2].toLong() and 0xFF) shl 16) or
               ((data[3].toLong() and 0xFF) shl 24) or
               ((data[4].toLong() and 0xFF) shl 32) or
               ((data[5].toLong() and 0xFF) shl 40) or
               ((data[6].toLong() and 0xFF) shl 48) or
               ((data[7].toLong() and 0xFF) shl 56)

        println("[DEBUG] Read header size: $headerSize")

        // Validate the header size
        if (headerSize <= 0 || headerSize > 1000000) { // 1MB is a reasonable upper limit for header size
            println("[DEBUG] Invalid header size: $headerSize")
            return 0
        }

        return headerSize
    }

    /**
     * Parses the JSON header string to extract tensor metadata.
     * 
     * The JSON header contains information about each tensor in the file,
     * including its data type, shape, and location within the file.
     * 
     * The format of the JSON header is:
     * ```json
     * {
     *   "tensor_name": {
     *     "dtype": "F32",
     *     "shape": [2, 3],
     *     "data_offsets": [0, 24]
     *   },
     *   ...
     * }
     * ```
     * 
     * @param headerJson The JSON header string.
     * @return A map of tensor names to their metadata.
     */
    private fun parseHeaderJson(headerJson: String): Map<String, StreamingTensorMetadata> {
        val result = mutableMapOf<String, StreamingTensorMetadata>()
        val jsonObject = Json.parseToJsonElement(headerJson).jsonObject

        // The safetensor format has a top-level object with tensor names as keys
        for ((name, tensorInfo) in jsonObject) {
            if (tensorInfo !is JsonObject) continue

            // Extract dtype
            val dtype = tensorInfo["dtype"]?.jsonPrimitive?.content ?: continue

            // Extract shape
            val shapeArray = tensorInfo["shape"]?.jsonArray ?: continue
            val shapeList = shapeArray.mapNotNull { it.jsonPrimitive.intOrNull }
            val shape = Shape(*shapeList.toIntArray())

            // Extract data_offsets
            val dataOffsetObj = tensorInfo["data_offsets"]?.jsonArray ?: continue
            if (dataOffsetObj.size != 2) continue
            val dataOffset = dataOffsetObj[0].jsonPrimitive.longOrNull ?: continue
            val dataEnd = dataOffsetObj[1].jsonPrimitive.longOrNull ?: continue
            val dataLength = dataEnd - dataOffset

            result[name] = StreamingTensorMetadata(dtype, shape, dataOffset, dataLength)
        }

        return result
    }

    /**
     * Metadata for a tensor in the safetensor file.
     * 
     * This class holds the information extracted from the JSON header for a specific tensor,
     * including its data type, shape, and location within the file.
     * 
     * @property dtype The data type of the tensor (e.g., "F32", "I8").
     * @property shape The shape of the tensor as a list of dimensions.
     * @property dataOffset The offset of the tensor data from the start of the data section.
     * @property dataLength The length of the tensor data in bytes.
     */
    data class StreamingTensorMetadata(
        val dtype: String,
        val shape: Shape,
        val dataOffset: Long,
        val dataLength: Long
    )
}
