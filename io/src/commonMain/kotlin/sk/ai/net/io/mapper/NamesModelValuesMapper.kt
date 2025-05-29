package sk.ai.net.io.mapper

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.nn.Module
import sk.ai.net.nn.reflection.ModuleParameters

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

class NamesBasedValuesModelMapper(
    private val matcher: (moduleParamName: String, wandbKey: String) -> Boolean = ::defaultNamesMatcher
) : ModelValuesMapper {

    override fun mapToModel(model: Module, wandb: Map<String, Tensor>) {
        traverseAndMap(model, wandb)
    }

    // Recursively traverse the module tree.
    private fun traverseAndMap(module: Module, wandb: Map<String, Tensor>) {
        if (module is ModuleParameters) {
            mapModuleParameters(module, module.name, wandb)
        }
        module.modules.forEach { child ->
            traverseAndMap(child, wandb)
        }
    }

    // For a module implementing ModuleParameters, match and update its parameters.
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
