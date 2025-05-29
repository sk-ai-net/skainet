package sk.ai.net.io.model

import kotlinx.io.Source
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.io.ParametersLoader

/**
 * A parameters loader that loads tensors from a model file using the ModelFormat abstraction.
 * 
 * This class allows loading tensors from different model formats (GGUF, SafeTensors, etc.)
 * in a format-agnostic way.
 * 
 * @property handleSource A function that returns a Source for the model file.
 */
class ModelFormatLoader(private val handleSource: () -> Source) : ParametersLoader {
    /**
     * Loads tensors from the model file and calls the onTensorLoaded callback for each tensor.
     * 
     * @param onTensorLoaded A callback function that is called for each tensor with the tensor name and the tensor itself.
     */
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        handleSource().use { source ->
            // Create a ModelFormat instance for the source
            val modelFormat = ModelFormat.create(source)
            
            // Get all tensors from the model file
            val tensors = modelFormat.getAllTensors()
            
            // Call the callback for each tensor
            tensors.forEach { (name, tensor) ->
                onTensorLoaded(name, tensor)
            }
        }
    }
    
    /**
     * Gets the metadata from the model file.
     * 
     * @return A map of metadata keys to values.
     */
    fun getMetadata(): Map<String, String> {
        return handleSource().use { source ->
            // Create a ModelFormat instance for the source
            val modelFormat = ModelFormat.create(source)
            
            // Get the metadata from the model file
            modelFormat.getMetadata()
        }
    }
    
    /**
     * Gets the names of all tensors in the model file.
     * 
     * @return A list of tensor names.
     */
    fun getTensorNames(): List<String> {
        return handleSource().use { source ->
            // Create a ModelFormat instance for the source
            val modelFormat = ModelFormat.create(source)
            
            // Get the tensor names from the model file
            modelFormat.getTensorNames()
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
            // Create a ModelFormat instance for the source
            val modelFormat = ModelFormat.create(source)
            
            // Get the tensor from the model file
            modelFormat.getTensor(name)
        }
    }
}