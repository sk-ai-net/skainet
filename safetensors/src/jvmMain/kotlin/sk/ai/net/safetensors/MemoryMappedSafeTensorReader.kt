package sk.ai.net.safetensors

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlinx.serialization.json.*
import sk.ai.net.graph.tensor.Int8Tensor
import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

/**
 * Implementation of SafeTensorReader that uses memory-mapped files for efficient loading of large safetensor files.
 *
 * This implementation uses memory-mapped files to avoid loading the entire file into memory at once.
 * It only maps the header and metadata into memory, and maps the tensor data into memory when requested.
 * This allows for efficient loading of large safetensor files, such as those used for LLMs.
 * 
 * This implementation can handle files larger than 2GB by mapping only the portions of the file that are needed,
 * rather than trying to map the entire file at once.
 *
 * @property file The file containing the safetensor data.
 * @property fileChannel The file channel for accessing the file.
 * @property headerBuffer The memory-mapped buffer for the header and metadata.
 * @property randomAccessFile The random access file for accessing the file.
 */
class MemoryMappedSafeTensorsReader private constructor(
    private val file: File,
    private val fileChannel: FileChannel,
    private val headerBuffer: MappedByteBuffer,
    private val randomAccessFile: RandomAccessFile
) : SafeTensorsReader {
    // Header information
    private val headerSize: Long
    private val metadata: Map<String, MappedTensorMetadata>

    init {
        // Parse the header
        headerSize = readHeaderSize(headerBuffer)
        metadata = parseHeader(headerBuffer, headerSize)
    }

    override fun readTensor(name: String): Tensor? {
        val meta = metadata[name] ?: return null
        return readTensorData(fileChannel, meta, headerSize)
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
     * @param buffer The buffer containing the safetensor data.
     * @return The size of the JSON header in bytes.
     */
    private fun readHeaderSize(buffer: ByteBuffer): Long {
        buffer.position(0)
        return buffer.order(ByteOrder.LITTLE_ENDIAN).getLong()
    }

    /**
     * Extracts and parses the JSON header from the safetensor file.
     * 
     * The header starts after the 8-byte header size and contains metadata
     * about all tensors stored in the file.
     * 
     * @param buffer The buffer containing the safetensor data.
     * @param headerSize The size of the JSON header in bytes.
     * @return A map of tensor names to their metadata.
     */
    private fun parseHeader(buffer: ByteBuffer, headerSize: Long): Map<String, MappedTensorMetadata> {
        // Extract the header JSON string
        val headerBytes = ByteArray(headerSize.toInt())
        buffer.position(8)
        buffer.get(headerBytes)
        val headerJson = headerBytes.decodeToString()

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
    private fun parseHeaderJson(headerJson: String): Map<String, MappedTensorMetadata> {
        val result = mutableMapOf<String, MappedTensorMetadata>()
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

            result[name] = MappedTensorMetadata(dtype, shape, dataOffset, dataLength)
        }

        return result
    }

    /**
     * Reads tensor data from the memory-mapped file and creates a Tensor object.
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
     * @param buffer The buffer containing the safetensor data.
     * @param metadata The metadata for the tensor to read.
     * @param headerSize The size of the JSON header in bytes.
     * @return A Tensor object containing the tensor data.
     */
    private fun readTensorData(fileChannel: FileChannel, metadata: MappedTensorMetadata, headerSize: Long): Tensor {
        // Calculate the absolute offset in the file (header size + JSON header + data offset)
        val absoluteOffset = 8 + headerSize + metadata.dataOffset
        val dataLength = metadata.dataLength

        // Create a tensor based on the dtype and shape
        return when (metadata.dtype) {
            "F32" -> {
                try {
                    // Map only the tensor data
                    val tensorBuffer = fileChannel.map(
                        FileChannel.MapMode.READ_ONLY,
                        absoluteOffset,
                        dataLength.coerceAtMost(Int.MAX_VALUE.toLong())
                    )
                    tensorBuffer.order(ByteOrder.LITTLE_ENDIAN)

                    // Convert to double array for F32 data
                    val elementCount = (dataLength / 4).toInt()
                    val doubleData = DoubleArray(elementCount)
                    for (i in doubleData.indices) {
                        doubleData[i] = tensorBuffer.getFloat().toDouble()
                    }
                    SimpleTensor(metadata.shape, doubleData)
                } catch (e: Exception) {
                    println("Error reading tensor data: ${e.message}")
                    // For errors, return a tensor with zeros
                    val totalElements = metadata.shape.volume
                    SimpleTensor(metadata.shape, DoubleArray(totalElements))
                }
            }
            "I8" -> {
                try {
                    // Map only the tensor data
                    val tensorBuffer = fileChannel.map(
                        FileChannel.MapMode.READ_ONLY,
                        absoluteOffset,
                        dataLength.coerceAtMost(Int.MAX_VALUE.toLong())
                    )

                    // Extract the tensor data from the buffer
                    val tensorData = ByteArray(dataLength.toInt())
                    tensorBuffer.get(tensorData)

                    // For I8 data, we can use Int8Tensor with appropriate scale and zero point
                    // For simplicity, we're using a default scale and zero point here
                    // In a real implementation, these would be extracted from the metadata
                    val scale = 1.0
                    val zeroPoint = 0
                    Int8Tensor(metadata.shape, tensorData, scale, zeroPoint)
                } catch (e: Exception) {
                    println("Error reading tensor data: ${e.message}")
                    // For errors, return a tensor with zeros
                    val totalElements = metadata.shape.volume
                    SimpleTensor(metadata.shape, DoubleArray(totalElements))
                }
            }
            else -> {
                // For unsupported dtypes, convert to double array with zeros
                val totalElements = metadata.shape.volume
                SimpleTensor(metadata.shape, DoubleArray(totalElements))
            }
        }
    }

    companion object {
        /**
         * Creates a MemoryMappedSafeTensorReader from the given file.
         *
         * @param file The file containing the safetensor data.
         * @return A MemoryMappedSafeTensorReader instance.
         */
        fun fromFile(file: File): MemoryMappedSafeTensorsReader {
            val randomAccessFile = RandomAccessFile(file, "r")
            val fileChannel = randomAccessFile.channel

            // Map only the header (first 8 bytes to get the header size)
            val headerSizeBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, 8)
            headerSizeBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val headerSize = headerSizeBuffer.getLong(0)

            // Map the header and metadata
            val headerBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                0,
                (8 + headerSize).coerceAtMost(Int.MAX_VALUE.toLong())
            )

            return MemoryMappedSafeTensorsReader(file, fileChannel, headerBuffer, randomAccessFile)
        }

        /**
         * Creates a MemoryMappedSafeTensorReader from the given file path.
         *
         * This implementation uses memory-mapped files for efficient loading of large safetensor files.
         * It only maps the header and metadata into memory, and maps the tensor data into memory when requested.
         * This allows for efficient loading of large safetensor files, such as those used for LLMs.
         *
         * @param filePath The path to the file containing the safetensor data.
         * @return A MemoryMappedSafeTensorReader instance.
         */
        fun fromFilePath(filePath: String): MemoryMappedSafeTensorsReader {
            return fromFile(File(filePath))
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
private data class MappedTensorMetadata(
    val dtype: String,
    val shape: Shape,
    val dataOffset: Long,
    val dataLength: Long
)
