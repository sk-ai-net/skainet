package sk.ai.net.graph.serialization

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.core.nn.Module
import sk.ai.net.core.tensor.Tensor
import java.io.File

/**
 * A JVM-specific implementation of the ModelSerializer interface.
 *
 * This class extends JsonModelSerializer to provide file operations for the JVM platform.
 *
 * @param T The type of data processed by the models.
 */
class JvmModelSerializer<T>(
    private val jsonSerializer: JsonModelSerializer<T>
) : ModelSerializer<T> {
    /**
     * Serializes a computation node to a string representation.
     *
     * @param node The computation node to serialize.
     * @return A string representation of the node.
     */
    override fun serializeNode(node: ComputeNode<T>): String {
        return jsonSerializer.serializeNode(node)
    }

    /**
     * Deserializes a string representation to a computation node.
     *
     * @param serialized The string representation of the node.
     * @return The deserialized computation node.
     */
    override fun deserializeNode(serialized: String): ComputeNode<T> {
        return jsonSerializer.deserializeNode(serialized)
    }

    /**
     * Serializes a module to a string representation.
     *
     * @param module The module to serialize.
     * @return A string representation of the module.
     */
    override fun serializeModule(module: Module<T>): String {
        return jsonSerializer.serializeModule(module)
    }

    /**
     * Deserializes a string representation to a module.
     *
     * @param serialized The string representation of the module.
     * @return The deserialized module.
     */
    override fun deserializeModule(serialized: String): Module<T> {
        return jsonSerializer.deserializeModule(serialized)
    }

    /**
     * Saves a computation node to a file.
     *
     * @param node The computation node to save.
     * @param filePath The path to the file where the node will be saved.
     */
    override fun saveNode(node: ComputeNode<T>, filePath: String) {
        val serialized = serializeNode(node)
        File(filePath).writeText(serialized)
    }

    /**
     * Loads a computation node from a file.
     *
     * @param filePath The path to the file containing the serialized node.
     * @return The loaded computation node.
     */
    override fun loadNode(filePath: String): ComputeNode<T> {
        val serialized = File(filePath).readText()
        return deserializeNode(serialized)
    }

    /**
     * Saves a module to a file.
     *
     * @param module The module to save.
     * @param filePath The path to the file where the module will be saved.
     */
    override fun saveModule(module: Module<T>, filePath: String) {
        val serialized = serializeModule(module)
        File(filePath).writeText(serialized)
    }

    /**
     * Loads a module from a file.
     *
     * @param filePath The path to the file containing the serialized module.
     * @return The loaded module.
     */
    override fun loadModule(filePath: String): Module<T> {
        val serialized = File(filePath).readText()
        return deserializeModule(serialized)
    }
}

/**
 * Creates a JVM model serializer for tensor data.
 *
 * @return A JVM model serializer for tensor data.
 */
fun createJvmTensorSerializer(): JvmModelSerializer<Tensor> {
    return JvmModelSerializer(createTensorJsonSerializer())
}