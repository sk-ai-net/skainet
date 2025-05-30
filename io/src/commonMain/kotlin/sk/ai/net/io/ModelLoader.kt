package sk.ai.net.io

import sk.ai.net.core.tensor.Tensor
import sk.ai.net.io.model.ModelFormatType

/**
 * Interface for loading model parameters from various sources.
 *
 * This interface provides a common abstraction for loading tensor parameters
 * from different storage formats (CSV, GGUF, SafeTensors, etc.) and making them
 * available to the model.
 */
interface ModelLoader {

    val modelFormatType: ModelFormatType

    /**
     * Loads parameters and invokes the provided callback for each tensor loaded.
     *
     * This method asynchronously loads tensors from the source and calls the
     * provided callback function for each tensor as it is loaded. This allows
     * for streaming large models without loading everything into memory at once.
     *
     * @param onTensorLoaded A callback function that is invoked for each tensor loaded.
     *                       The callback receives the tensor name and the tensor itself.
     */
    suspend fun load(onTensorLoaded: (String, Tensor) -> Unit)
}
