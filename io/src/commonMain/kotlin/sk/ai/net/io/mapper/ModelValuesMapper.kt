package sk.ai.net.io.mapper

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module

/**
 * Interface for mapping tensor values to a model's parameters.
 *
 * This interface provides functionality to map a collection of named tensors
 * to the appropriate parameters in a neural network model. It's used to load
 * pre-trained weights into a model structure.
 */
interface ModelValuesMapper {
    /**
     * Maps tensor values to the parameters of a model.
     *
     * This method takes a model and a map of tensor names to tensor values,
     * and assigns the tensor values to the corresponding parameters in the model.
     * The mapping between tensor names and model parameters is implementation-specific.
     *
     * @param model The model to map the values to.
     * @param wandb A map of tensor names to tensor values.
     */
    fun mapToModel(model: Module, wandb: Map<String, Tensor>)
}
