package sk.ai.net.io.model

import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.shape.Shape
import sk.ai.net.io.ModelLoader
import sk.ai.net.gguf.GGUFReader
import sk.ai.net.gguf.ReaderTensor
import sk.ai.net.safetensors.SafeTensorsReader

/**
 * Data class to hold tensor metadata.
 *
 * This class contains detailed information about a tensor in a model file,
 * including its name, shape, data type, and offset in the file.
 *
 * @property name The name of the tensor.
 * @property shape The shape of the tensor as a list of dimensions.
 * @property dataType The data type of the tensor (e.g., "F32", "I8", "Q4_0").
 * @property dataOffset The offset of the tensor data from the start of the data section.
 * @property dataLength The length of the tensor data in bytes.
 */
data class TensorMetadata(
    val name: String,
    val shape: List<Int>,
    val dataType: String,
    val dataOffset: Long,
    val dataLength: Long
)

/**
 * A parameters loader that loads tensors from a model file using the ModelFormat abstraction.
 *
 * This class allows loading tensors from different model formats (GGUF, SafeTensors, etc.)
 * in a format-agnostic way.
 *
 * Implementation follows these rules:
 * 1. We read from Source as a stream, without going back
 * 2. We read 2 bytes first to check for GGUF magic number
 * 3. Then we check first 8 bytes from SafeTensors header size
 * 4. After detecting file format, we read header/metadata
 * 5. Tensors are read after header/metadata
 *
 * @property handleSource A function that returns a Source for the model file.
 */
class ModelFormatLoader(private val handleSource: () -> Source) : ModelLoader {
    // Cache for model format type, initialized on first access
    private var cachedModelFormatType: ModelFormatType? = null

    override val modelFormatType: ModelFormatType
        get() {
            // Return cached value if available
            cachedModelFormatType?.let { return it }

            // Otherwise detect format
            return handleSource().use { source ->
                val formatType = detectModelFormat(source)
                cachedModelFormatType = formatType
                formatType
            }
        }

    /**
     * Loads tensors from the model file and calls the onTensorLoaded callback for each tensor.
     *
     * @param onTensorLoaded A callback function that is called for each tensor with the tensor name and the tensor itself.
     */
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        handleSource().use { source ->
            // Detect format and create appropriate model format handler
            when (detectModelFormat(source)) {
                ModelFormatType.GGUF -> {
                    // For GGUF, we need to read the entire file since GGUFReader requires it
                    val reader = GGUFReader(source)

                    // Process each tensor
                    reader.tensors.forEach { tensor ->
                        val convertedTensor = sk.ai.net.gguf.GGUFTensorConverter.convert(tensor)
                        onTensorLoaded(tensor.name, convertedTensor)
                    }
                }
                ModelFormatType.SAFETENSORS -> {
                    // For SafeTensors, use the streaming reader to avoid loading the entire file
                    val reader = SafeTensorsReader.fromSource { handleSource() }

                    // Get all tensor names and load each tensor
                    reader.getTensorNames().forEach { name ->
                        val tensor = reader.readTensor(name)
                        if (tensor != null) {
                            onTensorLoaded(name, tensor)
                        }
                    }
                }
                else -> throw IllegalArgumentException("Unsupported model format")
            }
        }
    }

    /**
     * Gets the metadata from the model file.
     *
     * @return A map of metadata keys to values.
     */
    fun getMetadata(): Map<String, String> {
        // First, detect the format type using a fresh source
        val formatType = modelFormatType
        println("[DEBUG] Detected format type: $formatType")

        // Then, use another fresh source to read the metadata
        return handleSource().use { source ->
            when (formatType) {
                ModelFormatType.GGUF -> {
                    // For GGUF, we need to read the entire file
                    val reader = GGUFReader(source)

                    // Print the available fields for debugging
                    println("[DEBUG] Available fields: ${reader.fields.keys}")

                    // Collect all string metadata fields
                    val metadata = reader.fields.keys
                        .mapNotNull { key -> 
                            val value = reader.getString(key)
                            println("[DEBUG] Field $key: $value")
                            if (value != null) key to value else null
                        }
                        .toMap()

                    println("[DEBUG] Extracted metadata: $metadata")
                    metadata
                }
                ModelFormatType.SAFETENSORS -> {
                    // Use the streaming reader to get metadata
                    val reader = SafeTensorsReader.fromSource { handleSource() }
                    val metadata = reader.getMetadata()
                    println("[DEBUG] SafeTensors metadata: $metadata")
                    metadata
                }
                else -> {
                    println("[DEBUG] Unknown format type")
                    emptyMap()
                }
            }
        }
    }

    /**
     * Gets the names of all tensors in the model file.
     *
     * @return A list of tensor names.
     */
    fun getTensorNames(): List<String> {
        return try {
            handleSource().use { source ->
                try {
                    when (detectModelFormat(source)) {
                        ModelFormatType.GGUF -> {
                            try {
                                // For GGUF, we need to read the entire file
                                val reader = GGUFReader(source)
                                reader.tensors.map { it.name }
                            } catch (e: Exception) {
                                println("[DEBUG] Error getting GGUF tensor names: ${e.message}")
                                emptyList()
                            }
                        }
                        ModelFormatType.SAFETENSORS -> {
                            try {
                                // Use the streaming reader to get tensor names
                                val reader = SafeTensorsReader.fromSource { handleSource() }
                                reader.getTensorNames()
                            } catch (e: Exception) {
                                println("[DEBUG] Error getting SafeTensors tensor names: ${e.message}")
                                emptyList()
                            }
                        }
                        else -> {
                            println("[DEBUG] Unknown model format")
                            emptyList()
                        }
                    }
                } catch (e: Exception) {
                    println("[DEBUG] Error detecting model format: ${e.message}")
                    emptyList()
                }
            }
        } catch (e: Exception) {
            println("[DEBUG] Error handling source: ${e.message}")
            emptyList()
        }
    }

    /**
     * Gets a specific tensor from the model file.
     *
     * @param name The name of the tensor to get.
     * @return The tensor, or null if not found.
     */
    fun getTensor(name: String): Tensor? {
        return handleSource().use { source ->
            when (detectModelFormat(source)) {
                ModelFormatType.GGUF -> {
                    // For GGUF, we need to read the entire file
                    val reader = GGUFReader(source)

                    // Find the tensor with the given name
                    val readerTensor = reader.tensors.find { it.name == name } ?: return null

                    // Convert it to a tensor
                    sk.ai.net.gguf.GGUFTensorConverter.convert(readerTensor)
                }
                ModelFormatType.SAFETENSORS -> {
                    // Use the streaming reader to get the tensor
                    val reader = SafeTensorsReader.fromSource { handleSource() }
                    reader.readTensor(name)
                }
                else -> null
            }
        }
    }

    /**
     * Gets detailed metadata for all tensors in the model file.
     *
     * This method returns detailed information about each tensor in the model file,
     * including its name, shape, data type, and offset in the file.
     *
     * @return A list of TensorMetadata objects, one for each tensor in the model file.
     */
    fun getTensorMetadata(): List<TensorMetadata> {
        return try {
            handleSource().use { source ->
                try {
                    when (detectModelFormat(source)) {
                        ModelFormatType.GGUF -> {
                            try {
                                // For GGUF, we need to read the entire file
                                val reader = GGUFReader(source)

                                // Convert ReaderTensor to TensorMetadata
                                reader.tensors.map { tensor ->
                                    TensorMetadata(
                                        name = tensor.name,
                                        shape = tensor.shape.map { it.toInt() },
                                        dataType = tensor.tensorType.toString(),
                                        dataOffset = tensor.dataOffset.toLong(),
                                        dataLength = tensor.nBytes.toLong()
                                    )
                                }
                            } catch (e: Exception) {
                                println("[DEBUG] Error getting GGUF tensor metadata: ${e.message}")
                                emptyList()
                            }
                        }
                        ModelFormatType.SAFETENSORS -> {
                            try {
                                // Use the streaming reader to get tensor metadata
                                val reader = SafeTensorsReader.fromSource { handleSource() }

                                // Get the header JSON to extract tensor metadata
                                val headerJson = reader.getHeaderJson()
                                if (headerJson.isNotEmpty()) {
                                    // Parse the header JSON to extract tensor metadata
                                    parseHeaderJsonForMetadata(headerJson)
                                } else {
                                    // If we can't get the header JSON, use a fallback approach
                                    val tensorNames = reader.getTensorNames()
                                    tensorNames.mapNotNull { name ->
                                        val tensor = reader.readTensor(name)
                                        if (tensor != null) {
                                            TensorMetadata(
                                                name = name,
                                                shape = tensor.shape.dimensions.toList(),
                                                dataType = "Unknown", // We don't have this information without the header
                                                dataOffset = -1, // We don't have this information without the header
                                                dataLength = -1 // We don't have this information without the header
                                            )
                                        } else {
                                            null
                                        }
                                    }
                                }
                            } catch (e: Exception) {
                                println("[DEBUG] Error getting SafeTensors tensor metadata: ${e.message}")
                                emptyList()
                            }
                        }
                        else -> {
                            println("[DEBUG] Unknown model format")
                            emptyList()
                        }
                    }
                } catch (e: Exception) {
                    println("[DEBUG] Error detecting model format: ${e.message}")
                    emptyList()
                }
            }
        } catch (e: Exception) {
            println("[DEBUG] Error handling source: ${e.message}")
            emptyList()
        }
    }

    /**
     * Parses the header JSON to extract tensor metadata.
     *
     * This method is used for SafeTensors files to extract tensor metadata from the header JSON.
     *
     * @param headerJson The header JSON string.
     * @return A list of TensorMetadata objects, one for each tensor in the header.
     */
    private fun parseHeaderJsonForMetadata(headerJson: String): List<TensorMetadata> {
        try {
            val result = mutableListOf<TensorMetadata>()
            val jsonObject = kotlinx.serialization.json.Json.parseToJsonElement(headerJson).jsonObject

            // The safetensor format has a top-level object with tensor names as keys
            for ((name, tensorInfo) in jsonObject) {
                if (tensorInfo !is kotlinx.serialization.json.JsonObject) continue

                // Extract dtype
                val dtype = tensorInfo["dtype"]?.jsonPrimitive?.content ?: continue

                // Extract shape
                val shapeArray = tensorInfo["shape"]?.jsonArray ?: continue
                val shapeList = shapeArray.mapNotNull { it.jsonPrimitive.intOrNull }

                // Extract data_offsets
                val dataOffsetObj = tensorInfo["data_offsets"]?.jsonArray ?: continue
                if (dataOffsetObj.size != 2) continue
                val dataOffset = dataOffsetObj[0].jsonPrimitive.longOrNull ?: continue
                val dataEnd = dataOffsetObj[1].jsonPrimitive.longOrNull ?: continue
                val dataLength = dataEnd - dataOffset

                result.add(
                    TensorMetadata(
                        name = name,
                        shape = shapeList,
                        dataType = dtype,
                        dataOffset = dataOffset,
                        dataLength = dataLength
                    )
                )
            }

            return result
        } catch (e: Exception) {
            println("[DEBUG] Error parsing header JSON for metadata: ${e.message}")
            return emptyList()
        }
    }

    /**
     * Detects the model format by reading the first few bytes of the source.
     * 
     * This method follows the rules:
     * 1. Read first 2 bytes to check for GGUF magic number
     * 2. Check first 8 bytes for SafeTensors header size
     * 
     * @param source The source to read from
     * @return The detected model format type
     * @throws IllegalArgumentException If the format is not supported
     */
    private fun detectModelFormat(source: Source): ModelFormatType {
        // Read first 8 bytes to check both formats
        val bytes = source.readByteArray(8)

        // Check for GGUF format (magic number: 0x47475546 or "GGUF" in ASCII)
        if (bytes.size >= 4 &&
            bytes[0].toInt() == 0x47 && // 'G'
            bytes[1].toInt() == 0x47 && // 'G'
            bytes[2].toInt() == 0x55 && // 'U'
            bytes[3].toInt() == 0x46    // 'F'
        ) {
            println("[DEBUG] Detected GGUF format")
            return ModelFormatType.GGUF
        }

        // Check for SafeTensors format (first 8 bytes contain header size)
        if (bytes.size == 8) {
            // Try to read the header size as a 64-bit little-endian unsigned integer
            val headerSize = (bytes[0].toLong() and 0xFF) or
                            ((bytes[1].toLong() and 0xFF) shl 8) or
                            ((bytes[2].toLong() and 0xFF) shl 16) or
                            ((bytes[3].toLong() and 0xFF) shl 24) or
                            ((bytes[4].toLong() and 0xFF) shl 32) or
                            ((bytes[5].toLong() and 0xFF) shl 40) or
                            ((bytes[6].toLong() and 0xFF) shl 48) or
                            ((bytes[7].toLong() and 0xFF) shl 56)

            // Validate the header size - it should be a reasonable value
            // SafeTensors headers are typically not extremely large
            if (headerSize > 0 && headerSize < 1000000) { // 1MB is a reasonable upper limit for header size
                println("[DEBUG] Detected SafeTensors format with header size: $headerSize")
                return ModelFormatType.SAFETENSORS
            } else {
                println("[DEBUG] Invalid SafeTensors header size: $headerSize")
            }
        }

        // If we can't detect the format, throw an exception
        throw IllegalArgumentException("Unsupported model format")
    }
}
