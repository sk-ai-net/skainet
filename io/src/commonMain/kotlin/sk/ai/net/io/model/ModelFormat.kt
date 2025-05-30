package sk.ai.net.io.model

import kotlinx.io.Source
import kotlinx.io.readByteArray
import sk.ai.net.core.tensor.Tensor

/**
 * Interface for reading model formats like GGUF and SafeTensors.
 *
 * This interface provides a common abstraction for reading different model formats,
 * allowing the application to work with models in a format-agnostic way.
 */
interface ModelFormat {

    val formatType: ModelFormatType

    /**
     * Gets the metadata from the model file.
     *
     * @return A map of metadata keys to values.
     */
    fun getMetadata(): Map<String, String>

    /**
     * Gets the names of all tensors in the model file.
     *
     * @return A list of tensor names.
     */
    fun getTensorNames(): List<String>

    /**
     * Gets a specific tensor from the model file.
     *
     * @param name The name of the tensor to get.
     * @return The tensor, or null if not found.
     */
    fun getTensor(name: String): Tensor?

    /**
     * Gets all tensors from the model file.
     *
     * @return A map of tensor names to tensors.
     */
    fun getAllTensors(): Map<String, Tensor>

    companion object {
        /**
         * Creates a ModelFormat instance for the given source.
         *
         * This method attempts to detect the format of the model file and create
         * the appropriate ModelFormat instance.
         *
         * @param source The source to read the model file from.
         * @return A ModelFormat instance for the given source.
         * @throws IllegalArgumentException If the format of the model file is not supported.
         */
        fun create(source: Source): ModelFormat {
            // Make a copy of the source so we can read from it multiple times
            val bytes = source.peek().readByteArray(8)

            // Try to detect the format based on the file's magic number or structure

            // Check for GGUF format (magic number: 0x46554747 or "GGUF" in ASCII)
            if (bytes.size >= 4 &&
                bytes[0].toInt() == 0x47 && // 'G'
                bytes[1].toInt() == 0x47 && // 'G'
                bytes[2].toInt() == 0x55 && // 'U'
                bytes[3].toInt() == 0x46
            ) { // 'F'
                // Create a new source from the bytes
                return GGUFModelFormat(source)
            }


            // Check for SafeTensors format (JSON header at the beginning)
            if (bytes.size >= 2 && bytes[0] == 0xF8.toByte() && bytes[1] == 0xC8.toByte()) { // '{' and '"'
                // This is a very simple heuristic; SafeTensors files start with a JSON object
                return SafeTensorsModelFormat(source)
            }

            // If we can't detect the format, throw an exception
            throw IllegalArgumentException("Unsupported model format")
        }
    }
}
