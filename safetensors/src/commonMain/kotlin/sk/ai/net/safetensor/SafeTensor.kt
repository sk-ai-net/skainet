package sk.ai.net.safetensor

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import sk.ai.net.graph.tensor.Int8Tensor
import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

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
interface SafeTensorReader {
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

    companion object {
        /**
         * Creates a SafeTensorReader from the given byte array.
         *
         * This method loads the entire safetensor data into memory. For large files,
         * consider using [fromFile] or [fromFilePath] on platforms that support it.
         *
         * @param data The byte array containing the safetensor data.
         * @return A SafeTensorReader instance.
         */
        fun fromByteArray(data: ByteArray): SafeTensorReader = SafeTensorReaderImpl(data)

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
        fun fromResource(resourcePath: String): SafeTensorReader {
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
        fun fromFilePath(filePath: String): SafeTensorReader {
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
        fun fromFile(file: Any): SafeTensorReader {
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
private class SafeTensorReaderImpl(private val data: ByteArray) : SafeTensorReader {
    // Header information
    private val headerSize: Long
    private val metadata: Map<String, TensorMetadata>

    init {
        // Parse the header
        headerSize = readHeaderSize(data)
        metadata = parseHeader(data, headerSize)
    }

    override fun readTensor(name: String): Tensor? {
        val meta = metadata[name] ?: return null
        return readTensorData(data, meta)
    }

    override fun getTensorNames(): List<String> {
        return metadata.keys.toList()
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
        // The first 8 bytes contain the header size as a 64-bit little-endian unsigned integer
        return (data[0].toLong() and 0xFF) or
               ((data[1].toLong() and 0xFF) shl 8) or
               ((data[2].toLong() and 0xFF) shl 16) or
               ((data[3].toLong() and 0xFF) shl 24) or
               ((data[4].toLong() and 0xFF) shl 32) or
               ((data[5].toLong() and 0xFF) shl 40) or
               ((data[6].toLong() and 0xFF) shl 48) or
               ((data[7].toLong() and 0xFF) shl 56)
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
        // Extract the header JSON string
        val headerJson = data.sliceArray(8 until (8 + headerSize).toInt()).decodeToString()

        // Parse the JSON header using kotlinx.serialization
        return parseHeaderJson(headerJson)
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

            result[name] = TensorMetadata(dtype, shape, dataOffset, dataLength)
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
     */
    private fun readTensorData(data: ByteArray, metadata: TensorMetadata): Tensor {
        // Calculate the absolute offset in the file (header size + JSON header + data offset)
        val absoluteOffset = 8 + headerSize + metadata.dataOffset
        val dataLength = metadata.dataLength.toInt()

        // Extract the tensor data from the file
        val tensorData = data.sliceArray(absoluteOffset.toInt() until (absoluteOffset.toInt() + dataLength))

        // Create a tensor based on the dtype and shape
        return when (metadata.dtype) {
            "F32" -> {
                // Convert byte array to double array for F32 data
                val doubleData = DoubleArray(tensorData.size / 4)
                for (i in doubleData.indices) {
                    val bytes = tensorData.sliceArray(i * 4 until (i * 4 + 4))
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
                // For unsupported dtypes, convert to double array with zeros
                val totalElements = metadata.shape.volume
                SimpleTensor(metadata.shape, DoubleArray(totalElements))
            }
        }
    }
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
private data class TensorMetadata(
    val dtype: String,
    val shape: Shape,
    val dataOffset: Long,
    val dataLength: Long
)

/**
 * Dummy tensor implementation for placeholder purposes.
 * 
 * This class provides a minimal implementation of the Tensor interface,
 * returning zero for all element accesses. It's used as a fallback when
 * a proper tensor implementation cannot be created.
 * 
 * @property shape The shape of the tensor as a list of dimensions.
 */
private class DummyTensor(override val shape: Shape) : Tensor {
    override fun get(vararg indices: Int): Double {
        return 0.0
    }
}
