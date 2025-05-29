package sk.ai.net.io.model

import kotlinx.io.Source
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.safetensors.SafeTensorsReader

/**
 * Implementation of ModelFormat for SafeTensors files.
 * 
 * This class uses the SafeTensorsReader to read SafeTensors files and provides
 * the data through the ModelFormat interface.
 * 
 * @property reader The SafeTensorsReader to use for reading the SafeTensors file.
 */
class SafeTensorsModelFormat(private val reader: SafeTensorsReader) : ModelFormat {
    /**
     * Constructor that creates a SafeTensorsReader from a Source.
     * 
     * @param source The source to read the SafeTensors file from.
     */
    constructor(source: Source) : this(SafeTensorsReader.fromByteArray(source.readBytes()))
    
    /**
     * Gets the metadata from the SafeTensors file.
     * 
     * Note: SafeTensors files don't have a standard metadata format like GGUF files,
     * so this method returns an empty map.
     * 
     * @return An empty map.
     */
    override fun getMetadata(): Map<String, String> {
        // SafeTensors files don't have a standard metadata format like GGUF files
        return emptyMap()
    }
    
    /**
     * Gets the names of all tensors in the SafeTensors file.
     * 
     * @return A list of tensor names.
     */
    override fun getTensorNames(): List<String> {
        return reader.getTensorNames()
    }
    
    /**
     * Gets a specific tensor from the SafeTensors file.
     * 
     * @param name The name of the tensor to get.
     * @return The tensor, or null if not found.
     */
    override fun getTensor(name: String): Tensor? {
        return reader.readTensor(name)
    }
    
    /**
     * Gets all tensors from the SafeTensors file.
     * 
     * @return A map of tensor names to tensors.
     */
    override fun getAllTensors(): Map<String, Tensor> {
        val tensorNames = reader.getTensorNames()
        return tensorNames.mapNotNull { name ->
            val tensor = reader.readTensor(name)
            if (tensor != null) name to tensor else null
        }.toMap()
    }
}