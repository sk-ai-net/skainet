package sk.ai.net.graph.serialization

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.core.ValueNode
import sk.ai.net.graph.core.AddNode
import sk.ai.net.graph.core.MultiplyNode
import sk.ai.net.graph.core.ActivationNode
import sk.ai.net.graph.nn.Module
import sk.ai.net.graph.nn.Sequential
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.SimpleTensor

/**
 * A JSON-based implementation of the ModelSerializer interface.
 *
 * This class serializes and deserializes computation models to and from JSON format.
 * It supports the basic types of computation nodes and modules.
 *
 * @param T The type of data processed by the models.
 * @property serializeValue A function that serializes a value of type T to a string.
 * @property deserializeValue A function that deserializes a string to a value of type T.
 */
class JsonModelSerializer<T>(
    private val serializeValue: (T) -> String,
    private val deserializeValue: (String) -> T,
    private val createAddNode: ((T, T) -> T) -> AddNode<T>,
    private val createMultiplyNode: ((T, T) -> T) -> MultiplyNode<T>,
    private val createActivationNode: ((T) -> T, String) -> ActivationNode<T>
) : ModelSerializer<T> {
    /**
     * Serializes a computation node to a JSON string.
     *
     * @param node The computation node to serialize.
     * @return A JSON string representation of the node.
     */
    override fun serializeNode(node: ComputeNode<T>): String {
        val json = StringBuilder()
        json.append("{")

        when (node) {
            is ValueNode<*> -> {
                json.append("\"type\":\"ValueNode\",")
                json.append("\"value\":\"${serializeValue(node.evaluate())}\"")
            }
            is AddNode<*> -> {
                json.append("\"type\":\"AddNode\",")
                json.append("\"inputs\":[")
                node.inputs.forEachIndexed { index, input ->
                    if (index > 0) json.append(",")
                    json.append(serializeNode(input))
                }
                json.append("]")
            }
            is MultiplyNode<*> -> {
                json.append("\"type\":\"MultiplyNode\",")
                json.append("\"inputs\":[")
                node.inputs.forEachIndexed { index, input ->
                    if (index > 0) json.append(",")
                    json.append(serializeNode(input))
                }
                json.append("]")
            }
            is ActivationNode<*> -> {
                json.append("\"type\":\"ActivationNode\",")
                json.append("\"name\":\"${node}\",")
                json.append("\"inputs\":[")
                node.inputs.forEachIndexed { index, input ->
                    if (index > 0) json.append(",")
                    json.append(serializeNode(input))
                }
                json.append("]")
            }
            else -> {
                throw IllegalArgumentException("Unsupported node type: ${node::class.simpleName}")
            }
        }

        json.append("}")
        return json.toString()
    }

    /**
     * Deserializes a JSON string to a computation node.
     *
     * @param serialized The JSON string representation of the node.
     * @return The deserialized computation node.
     */
    override fun deserializeNode(serialized: String): ComputeNode<T> {
        // This is a simplified implementation that doesn't handle all edge cases
        val type = extractJsonValue(serialized, "type")

        return when (type) {
            "ValueNode" -> {
                val value = extractJsonValue(serialized, "value")
                ValueNode(deserializeValue(value))
            }
            "AddNode" -> {
                val inputs = extractJsonArray(serialized, "inputs")
                val node = createAddNode { a, b -> a } // Placeholder function
                inputs.forEach { input ->
                    node.inputs.add(deserializeNode(input))
                }
                node
            }
            "MultiplyNode" -> {
                val inputs = extractJsonArray(serialized, "inputs")
                val node = createMultiplyNode { a, b -> a } // Placeholder function
                inputs.forEach { input ->
                    node.inputs.add(deserializeNode(input))
                }
                node
            }
            "ActivationNode" -> {
                val inputs = extractJsonArray(serialized, "inputs")
                val name = extractJsonValue(serialized, "name")
                val node = createActivationNode({ it }, name) // Placeholder function
                inputs.forEach { input ->
                    node.inputs.add(deserializeNode(input))
                }
                node
            }
            else -> {
                throw IllegalArgumentException("Unsupported node type: $type")
            }
        }
    }

    /**
     * Serializes a module to a JSON string.
     *
     * @param module The module to serialize.
     * @return A JSON string representation of the module.
     */
    override fun serializeModule(module: Module<T>): String {
        val json = StringBuilder()
        json.append("{")

        when (module) {
            is Sequential<*> -> {
                json.append("\"type\":\"Sequential\",")
                json.append("\"modules\":[")
                module.modules.forEachIndexed { index, subModule ->
                    if (index > 0) json.append(",")
                    json.append(serializeModule(subModule as Module<T>))
                }
                json.append("]")
            }
            else -> {
                throw IllegalArgumentException("Unsupported module type: ${module::class.simpleName}")
            }
        }

        json.append("}")
        return json.toString()
    }

    /**
     * Deserializes a JSON string to a module.
     *
     * @param serialized The JSON string representation of the module.
     * @return The deserialized module.
     */
    override fun deserializeModule(serialized: String): Module<T> {
        // This is a simplified implementation that doesn't handle all edge cases
        val type = extractJsonValue(serialized, "type")

        return when (type) {
            "Sequential" -> {
                val moduleStrings = extractJsonArray(serialized, "modules")
                val modules = moduleStrings.map { deserializeModule(it) }.toTypedArray()
                Sequential(*modules)
            }
            else -> {
                throw IllegalArgumentException("Unsupported module type: $type")
            }
        }
    }

    /**
     * Saves a computation node to a file.
     *
     * @param node The computation node to save.
     * @param filePath The path to the file where the node will be saved.
     */
    override fun saveNode(node: ComputeNode<T>, filePath: String) {
        // Platform-specific file writing would be implemented here
        // For now, we'll just throw an exception
        throw UnsupportedOperationException("File operations are not supported in common code")
    }

    /**
     * Loads a computation node from a file.
     *
     * @param filePath The path to the file containing the serialized node.
     * @return The loaded computation node.
     */
    override fun loadNode(filePath: String): ComputeNode<T> {
        // Platform-specific file reading would be implemented here
        // For now, we'll just throw an exception
        throw UnsupportedOperationException("File operations are not supported in common code")
    }

    /**
     * Saves a module to a file.
     *
     * @param module The module to save.
     * @param filePath The path to the file where the module will be saved.
     */
    override fun saveModule(module: Module<T>, filePath: String) {
        // Platform-specific file writing would be implemented here
        // For now, we'll just throw an exception
        throw UnsupportedOperationException("File operations are not supported in common code")
    }

    /**
     * Loads a module from a file.
     *
     * @param filePath The path to the file containing the serialized module.
     * @return The loaded module.
     */
    override fun loadModule(filePath: String): Module<T> {
        // Platform-specific file reading would be implemented here
        // For now, we'll just throw an exception
        throw UnsupportedOperationException("File operations are not supported in common code")
    }

    /**
     * Extracts a value from a JSON string.
     *
     * @param json The JSON string.
     * @param key The key of the value to extract.
     * @return The extracted value.
     */
    private fun extractJsonValue(json: String, key: String): String {
        val regex = "\"$key\":\"([^\"]+)\"".toRegex()
        val matchResult = regex.find(json) ?: throw IllegalArgumentException("Key not found: $key")
        return matchResult.groupValues[1]
    }

    /**
     * Extracts an array from a JSON string.
     *
     * @param json The JSON string.
     * @param key The key of the array to extract.
     * @return The extracted array as a list of strings.
     */
    private fun extractJsonArray(json: String, key: String): List<String> {
        val regex = "\"$key\":\\[(.+)\\]".toRegex()
        val matchResult = regex.find(json) ?: throw IllegalArgumentException("Key not found: $key")
        val arrayContent = matchResult.groupValues[1]

        // This is a simplified implementation that doesn't handle nested arrays properly
        val result = mutableListOf<String>()
        var depth = 0
        var start = 0

        for (i in arrayContent.indices) {
            when (arrayContent[i]) {
                '{' -> depth++
                '}' -> depth--
                ',' -> if (depth == 0) {
                    result.add(arrayContent.substring(start, i))
                    start = i + 1
                }
            }
        }

        if (start < arrayContent.length) {
            result.add(arrayContent.substring(start))
        }

        return result
    }
}

/**
 * Creates a JSON model serializer for tensor data.
 *
 * @return A JSON model serializer for tensor data.
 */
fun createTensorJsonSerializer(): JsonModelSerializer<Tensor> {
    return JsonModelSerializer(
        serializeValue = { tensor ->
            // Simplified serialization for tensors
            val shape = tensor.shape.dimensions.joinToString(",")
            val values = (0 until tensor.shape.volume).joinToString(",") { i ->
                val indices = mutableListOf<Int>()
                var remaining = i
                for (dim in tensor.shape.dimensions) {
                    indices.add(remaining % dim)
                    remaining /= dim
                }
                tensor.get(*indices.toIntArray()).toString()
            }
            "$shape|$values"
        },
        deserializeValue = { serialized ->
            // Simplified deserialization for tensors
            val parts = serialized.split("|")
            val dimensions = parts[0].split(",").map { it.toInt() }.toIntArray()
            val values = parts[1].split(",").map { it.toDouble() }.toDoubleArray()
            SimpleTensor(sk.ai.net.graph.tensor.shape.Shape(dimensions), values)
        },
        createAddNode = { add -> AddNode(add) },
        createMultiplyNode = { multiply -> MultiplyNode(multiply) },
        createActivationNode = { activate, name -> ActivationNode(activate, name) }
    )
}
