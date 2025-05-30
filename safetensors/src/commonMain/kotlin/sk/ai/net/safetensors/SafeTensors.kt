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
 * Implementation of the Hugging Face safetensor file format.
 *
 * The safetensor format is a binary format for storing tensors, designed for efficient
 * and safe loading of machine learning models. It consists of:
 * 1. An 8-byte header size (64-bit little-endian unsigned integer)
 * 2. A JSON header containing metadata about the tensors (size specified by the header size)
 * 3. The binary tensor data (with offsets specified in the JSON header)
 *
 * The format is designed to be memory-mapped efficiently and to prevent deserialization
 * vulnerabilities common in other formats. It allows for fast random access to individual
 * tensors without loading the entire file into memory.
 *
 * For a visual representation of the format, see the diagram at:
 * docs/images/safetensors-format.svg
 *
 * This interface provides functionality to read tensors from safetensor files.
 *
 * @see <a href="https://huggingface.co/docs/safetensors/index">Hugging Face Safetensors Documentation</a>
 */
interface SafeTensorsReader {
    /**
     * Reads a tensor from the safetensor data.
     *
     * @param name The name of the tensor to read.
     * @return The tensor if found, or null if not found.
     */
    fun readTensor(name: String): Tensor?

    /**
     * Gets all tensor names available in the safetensor file.
     *
     * @return A list of tensor names.
     */
    fun getTensorNames(): List<String>

    /**
     * Gets the metadata from the safetensor file.
     *
     * @return A map of metadata keys to values.
     */
    fun getMetadata(): Map<String, String>

    /**
     * Gets the JSON header from the safetensor file.
     *
     * This method returns the raw JSON header string from the safetensor file.
     * The header contains metadata about all tensors in the file, including their
     * data types, shapes, and offsets in the file.
     *
     * @return The JSON header string.
     */
    fun getHeaderJson(): String

    companion object Companion {
        /**
         * Creates a SafeTensorReader from the given byte array.
         *
         * This method loads the entire safetensor data into memory. For large files,
         * consider using [fromFile], [fromFilePath], or [fromSource] on platforms that support it.
         *
         * @param data The byte array containing the safetensor data.
         * @return A SafeTensorReader instance.
         */
        fun fromByteArray(data: ByteArray): SafeTensorsReader = SafeTensorsReaderImpl(data)

        /**
         * Creates a SafeTensorReader from the given source provider.
         *
         * This method reads only the header from the source and then reads tensor data on demand.
         * This is more efficient for large files than loading the entire file into memory.
         *
         * @param sourceProvider A function that returns a fresh Source each time it's called.
         * @return A SafeTensorReader instance.
         */
        fun fromSource(sourceProvider: () -> Source): SafeTensorsReader = StreamingSafeTensorsReader(sourceProvider)

        /**
         * Creates a SafeTensorReader from a resource file.
         *
         * This method loads a resource file from the classpath using platform-specific
         * resource loading mechanisms. The entire file is loaded into memory.
         * For large files, consider using [fromFile] or [fromFilePath] on platforms that support it.
         *
         * @param resourcePath The path to the resource file.
         * @return A SafeTensorReader instance.
         * @throws IllegalArgumentException If the resource could not be found.
         */
        fun fromResource(resourcePath: String): SafeTensorsReader {
            val resourceBytes = ResourceUtils.loadResourceAsBytes(resourcePath)
                ?: throw IllegalArgumentException("Resource not found: $resourcePath")

            return fromByteArray(resourceBytes)
        }

        /**
         * Creates a SafeTensorReader from the given file path.
         *
         * This method is platform-specific and may use memory-mapped files for efficient loading
         * of large safetensor files on platforms that support it.
         *
         * On platforms that don't support file operations, this method falls back to loading
         * the file as a resource.
         *
         * @param filePath The path to the file containing the safetensor data.
         * @return A SafeTensorReader instance.
         * @throws IllegalArgumentException If the file could not be found or read.
         */
        fun fromFilePath(filePath: String): SafeTensorsReader {
            // Default implementation falls back to loading as a resource
            // Platform-specific implementations will override this behavior
            return fromResource(filePath)
        }

        /**
         * Creates a SafeTensorReader from the given file.
         *
         * This method is platform-specific and may use memory-mapped files for efficient loading
         * of large safetensor files on platforms that support it.
         *
         * On platforms that don't support file operations, this method falls back to loading
         * the file as a resource.
         *
         * @param file The file containing the safetensor data.
         * @return A SafeTensorReader instance.
         * @throws IllegalArgumentException If the file could not be found or read.
         */
        fun fromFile(file: Any): SafeTensorsReader {
            // Default implementation falls back to loading as a resource
            // Platform-specific implementations will override this behavior
            return fromResource(file.toString())
        }
    }
}

/**
 * Implementation of SafeTensorReader that parses and reads safetensor files.
 * 
 * This implementation follows the safetensor format specification:
 * 1. Reads the 8-byte header size
 * 2. Parses the JSON header to extract tensor metadata
 * 3. Provides access to individual tensors based on their offsets in the file
 * 
 * The implementation supports different tensor data types (F32, I8) and
 * converts them to the appropriate tensor implementation.
 */
private class SafeTensorsReaderImpl(private val data: ByteArray) : SafeTensorsReader {
    // Header information
    private val headerSize: Long
    private val metadata: Map<String, TensorMetadata>
    private val headerJson: String

    init {
        // Parse the header
        headerSize = readHeaderSize(data)
        val headerEnd = (8 + headerSize).coerceAtMost(data.size.toLong()).toInt()
        headerJson = data.sliceArray(8 until headerEnd).decodeToString()
        metadata = parseHeaderJson(headerJson)
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

    override fun readTensor(name: String): Tensor? {
        val meta = metadata[name] ?: return null
        try {
            return readTensorData(data, meta)
        } catch (e: Exception) {
            println("[DEBUG] Error reading tensor $name: ${e.message}")
            return null
        }
    }

    override fun getTensorNames(): List<String> {
        try {
            return metadata.keys.toList()
        } catch (e: Exception) {
            println("[DEBUG] Error getting tensor names: ${e.message}")
            return emptyList()
        }
    }

    /**
     * Reads the header size from the first 8 bytes of the safetensor file.
     * 
     * The header size is stored as a 64-bit little-endian unsigned integer
     * at the beginning of the file.
     * 
     * @param data The byte array containing the safetensor data.
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
     * Extracts and parses the JSON header from the safetensor file.
     * 
     * The header starts after the 8-byte header size and contains metadata
     * about all tensors stored in the file.
     * 
     * @param data The byte array containing the safetensor data.
     * @param headerSize The size of the JSON header in bytes.
     * @return A map of tensor names to their metadata.
     */
    private fun parseHeader(data: ByteArray, headerSize: Long): Map<String, TensorMetadata> {
        // Validate header size to prevent IndexOutOfBoundsException
        if (headerSize <= 0 || headerSize > Int.MAX_VALUE) {
            println("[DEBUG] Invalid header size: $headerSize")
            return emptyMap()
        }

        // Ensure we don't try to read beyond the data array
        val headerEnd = (8 + headerSize).coerceAtMost(data.size.toLong()).toInt()
        if (headerEnd <= 8) {
            println("[DEBUG] Header end position ($headerEnd) is not after header size position (8)")
            return emptyMap()
        }

        try {
            // Extract the header JSON string
            val headerJson = data.sliceArray(8 until headerEnd).decodeToString()

            // Parse the JSON header using kotlinx.serialization
            return parseHeaderJson(headerJson)
        } catch (e: Exception) {
            println("[DEBUG] Error parsing SafeTensors header: ${e.message}")
            return emptyMap()
        }
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
    private fun parseHeaderJson(headerJson: String): Map<String, TensorMetadata> {
        val result = mutableMapOf<String, TensorMetadata>()
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

            result[name] = TensorMetadata(dtype, shape.dimensions.toList(), dataOffset, dataLength)
        }

        return result
    }

    /**
     * Reads tensor data from the file and creates a Tensor object.
     * 
     * This method extracts the binary data for a specific tensor from the file
     * based on its metadata, and converts it to the appropriate Tensor implementation
     * based on the data type.
     * 
     * Supported data types:
     * - F32: 32-bit floating point values, converted to SimpleTensor
     * - I8: 8-bit integer values, converted to Int8Tensor
     * 
     * For unsupported data types, a SimpleTensor with zeros is returned.
     * 
     * @param data The byte array containing the safetensor data.
     * @param metadata The metadata for the tensor to read.
     * @return A Tensor object containing the tensor data.
     * @throws Exception If there's an error reading the tensor data.
     */
    private fun readTensorData(data: ByteArray, metadata: TensorMetadata): Tensor {
        try {
            // Calculate the absolute offset in the file (header size + JSON header + data offset)
            val absoluteOffset = 8 + headerSize + metadata.dataOffset
            val dataLength = metadata.dataLength.toInt()

            // Validate the offset and length
            if (absoluteOffset < 0 || absoluteOffset >= data.size.toLong()) {
                println("[DEBUG] Invalid tensor offset: $absoluteOffset (data size: ${data.size})")
                throw IllegalArgumentException("Invalid tensor offset: $absoluteOffset")
            }

            if (dataLength <= 0 || absoluteOffset.toInt() + dataLength > data.size) {
                println("[DEBUG] Invalid tensor length: $dataLength (data size: ${data.size}, offset: $absoluteOffset)")
                throw IllegalArgumentException("Invalid tensor length: $dataLength")
            }

            // Extract the tensor data from the file
            val tensorData = data.sliceArray(absoluteOffset.toInt() until (absoluteOffset.toInt() + dataLength))

            // Create a tensor based on the dtype and shape
            return when (metadata.dtype) {
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
                    SimpleTensor(metadata.shape, doubleData)
                }
                "I8" -> {
                    // For I8 data, we can use Int8Tensor with appropriate scale and zero point
                    // For simplicity, we're using a default scale and zero point here
                    // In a real implementation, these would be extracted from the metadata
                    val scale = 1.0
                    val zeroPoint = 0
                    Int8Tensor(metadata.shape, tensorData, scale, zeroPoint)
                }
                else -> {
                    println("[DEBUG] Unsupported tensor dtype: ${metadata.dtype}")
                    // For unsupported dtypes, convert to double array with zeros
                    val totalElements = metadata.shape.fold(1) { acc, dim -> acc * dim }
                    SimpleTensor(metadata.shape, DoubleArray(totalElements))
                }
            }
        } catch (e: Exception) {
            println("[DEBUG] Error reading tensor data: ${e.message}")
            throw e
        }
    }
}
