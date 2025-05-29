package sk.ai.net.io.mapper

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module
import sk.ai.net.nn.reflection.ModuleParameters

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
    override fun mapToModel(model: Module, wandb: Map<String, Tensor>) {
        traverseAndMap(model, wandb)
    }

    /**
     * Recursively traverses the module tree and maps parameters.
     *
     * This method visits each module in the hierarchy, maps its parameters if it implements
     * [ModuleParameters], and then recursively processes its child modules.
     *
     * @param module The current module to process.
     * @param wandb A map of tensor names to tensor values.
     */
    private fun traverseAndMap(module: Module, wandb: Map<String, Tensor>) {
        if (module is ModuleParameters) {
            mapModuleParameters(module, module.name, wandb)
        }
        module.modules.forEach { child ->
            traverseAndMap(child, wandb)
        }
    }

    /**
     * Maps parameters for a module that implements [ModuleParameters].
     *
     * This method iterates through the parameters of the module and updates them
     * when a matching key is found in the weights and biases map.
     *
     * @param moduleParameters The module parameters to update.
     * @param moduleName The name of the module.
     * @param wandb A map of tensor names to tensor values.
     */
    private fun mapModuleParameters(
        moduleParameters: ModuleParameters,
        moduleName: String,
        wandb: Map<String, Tensor>
    ) {
        moduleParameters.params.forEach { param ->
            // Use the injected matcher function to find a matching wandb key.
            val matchingEntry = wandb.entries.find { (key, _) ->
                matcher(param.name, key)
            }
            if (matchingEntry != null) {
                param.value = matchingEntry.value
            }
        }
    }
}
