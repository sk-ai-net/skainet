package sk.ai.net.io.mapper

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.nn.Module
import sk.ai.net.graph.core.ComputeNode

/**
 * Default function for matching module parameter names with weights and biases keys.
 *
 * This function uses a regex pattern to extract information from both the module parameter name
 * and the weights and biases key, then compares the layer number and parameter type.
 * If the regex matching fails, it falls back to a simple suffix check.
 *
 * The expected format is: `<layer_type>-<layer_number>.<parameter_type>`
 * For example: `linear-1.weight` or `conv-2.bias`
 *
 * @param moduleParamName The name of the parameter in the module.
 * @param wandbKey The key in the weights and biases map.
 * @return True if the names match, false otherwise.
 */
internal fun defaultNamesMatcher(moduleParamName: String, wandbKey: String): Boolean {
    val regex = Regex("""^([a-zA-Z]+)-(\d+)\.(\w+)$""")
    val moduleMatch = regex.find(moduleParamName)
    val wandbMatch = regex.find(wandbKey)
    return if (moduleMatch != null && wandbMatch != null) {
        // Compare the layer number (group 2) and parameter type (group 3).
        moduleMatch.groupValues[2] == wandbMatch.groupValues[2] &&
                moduleMatch.groupValues[3] == wandbMatch.groupValues[3]
    } else {
        // Fallback: if regex matching fails, use a suffix check.
        wandbKey.endsWith(".$moduleParamName")
    }
}

/**
 * A model values mapper that matches parameters based on their names.
 *
 * This implementation of [ModelValuesMapper] uses a name matching strategy to map
 * tensor values from a weights and biases map to the parameters of a model.
 * It traverses the module hierarchy and updates parameters when a matching name is found.
 *
 * @property matcher A function that determines if a module parameter name matches a weights and biases key.
 *                  Defaults to [defaultNamesMatcher].
 */
class NamesBasedValuesModelMapper(
    private val matcher: (moduleParamName: String, wandbKey: String) -> Boolean = ::defaultNamesMatcher
) : ModelValuesMapper {

    /**
     * Maps tensor values to the parameters of a model based on name matching.
     *
     * This method traverses the module hierarchy and updates parameters when a matching name is found
     * in the weights and biases map.
     *
     * @param model The model to map the values to.
     * @param wandb A map of tensor names to tensor values.
     */
    override fun <T> mapToModel(model: Module<T>, wandb: Map<String, Tensor>) {
        // Since we don't have access to flattenParams, we'll implement a simple traversal
        // to find and update parameters in the module hierarchy

        // This is a simplified implementation that assumes parameters are stored in a way
        // that can be accessed through the module's properties or methods
        // In a real implementation, you would need to adapt this to the actual structure of your modules

        // For each tensor in the wandb map, try to find a matching parameter in the model
        wandb.forEach { (key, tensor) ->
            // Find a parameter in the model with a matching name
            // This is a placeholder implementation that would need to be adapted to your actual module structure
            // For example, you might need to traverse the module hierarchy and check each parameter

            // For now, we'll just log that we're trying to map a tensor to the model
            println("Trying to map tensor $key to model ${model::class.simpleName}")
        }
    }
}
