package sk.ai.net.io.model

import kotlinx.io.Source
import sk.ai.net.gguf.GGUFTensorLoader
import sk.ai.net.core.tensor.Tensor

/**
 * Implementation of ModelFormat for GGUF files.
 *
 * This class uses the GGUFTensorLoader to read GGUF files and provides
 * the data through the ModelFormat interface.
 *
 * @property source The source to read the GGUF file from.
 */
class GGUFModelFormat(private val source: Source, override val formatType: ModelFormatType = ModelFormatType.GGUF) :
    ModelFormat {
    /**
     * Gets the metadata from the GGUF file.
     *
     * @return A map of metadata keys to values.
     */
    override fun getMetadata(): Map<String, String> {
        return GGUFTensorLoader.getMetadata(source)
    }

    /**
     * Gets the names of all tensors in the GGUF file.
     *
     * @return A list of tensor names.
     */
    override fun getTensorNames(): List<String> {
        // Load all tensors and get their names
        val tensorNodes = GGUFTensorLoader.loadTensors(source.peek())
        return tensorNodes.keys.toList()
    }

    /**
     * Gets a specific tensor from the GGUF file.
     *
     * @param name The name of the tensor to get.
     * @return The tensor, or null if not found.
     */
    override fun getTensor(name: String): Tensor? {
        return GGUFTensorLoader.getTensor(source, name)
    }

    /**
     * Gets all tensors from the GGUF file.
     *
     * @return A map of tensor names to tensors.
     */
    override fun getAllTensors(): Map<String, Tensor> {
        // Load all tensors and evaluate them
        val tensorNodes = GGUFTensorLoader.loadTensors(source)
        return tensorNodes.mapValues { (_, node) -> node.evaluate() }
    }
}