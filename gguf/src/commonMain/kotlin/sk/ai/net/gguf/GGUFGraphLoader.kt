package sk.ai.net.gguf

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.TensorValueNode

/**
 * Utility class for loading GGUF files into the tensor-based neural network engine.
 * 
 * This class provides methods to load tensors from GGUF files and convert them to
 * the tensor format used by the neural network engine.
 * 
 * Note: The "Graph" in the name refers to the neural network computation graph architecture
 * that the tensors will be used in, not that GGUF files themselves contain graphs.
 * GGUF (GPT-Generated Unified Format) files contain tensor weights and metadata for
 * neural network models, which can then be used in a computation graph.
 */
object GGUFTensorLoader {

    /**
     * Loads tensors from a GGUF file and creates tensor value nodes for them.
     *
     * @param source The source to read the GGUF file from.
     * @return A map of tensor names to tensor value nodes.
     */
    fun loadTensors(source: Source): Map<String, TensorValueNode> {
        val reader = GGUFReader(source)

        // Convert each tensor to a graph tensor and create a value node for it
        return reader.tensors.associate { readerTensor ->
            val tensor = GGUFTensorConverter.convert(readerTensor)
            val valueNode = TensorValueNode(tensor)
            readerTensor.name to valueNode
        }
    }

    /**
     * Gets metadata from a GGUF file.
     *
     * @param source The source to read the GGUF file from.
     * @return A map of metadata keys to values.
     */
    fun getMetadata(source: Source): Map<String, String> {
        val reader = GGUFReader(source.peek())

        // Collect all string metadata fields
        return reader.fields.keys
            .mapNotNull { key -> 
                val value = reader.getString(key)
                if (value != null) key to value else null
            }
            .toMap()
    }

    /**
     * Gets a specific tensor from a GGUF file.
     *
     * @param source The source to read the GGUF file from.
     * @param name The name of the tensor to get.
     * @return The tensor, or null if not found.
     */
    fun getTensor(source: Source, name: String): Tensor? {
        val reader = GGUFReader(source.peek())

        // Find the tensor with the given name
        val readerTensor = reader.tensors.find { it.name == name } ?: return null

        // Convert it to a graph tensor
        return GGUFTensorConverter.convert(readerTensor)
    }
}
