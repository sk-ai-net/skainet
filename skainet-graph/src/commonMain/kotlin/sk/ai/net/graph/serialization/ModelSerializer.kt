package sk.ai.net.graph.serialization

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.core.nn.Module

/**
 * Interface for serializing and deserializing computation models.
 *
 * This interface defines methods for saving models to a string representation
 * and loading them back. It supports serializing both raw computation graphs
 * (ComputeNode) and higher-level neural network modules (Module).
 *
 * @param T The type of data processed by the models.
 */
interface ModelSerializer<T> {
    /**
     * Serializes a computation node to a string representation.
     *
     * @param node The computation node to serialize.
     * @return A string representation of the node.
     */
    fun serializeNode(node: ComputeNode<T>): String

    /**
     * Deserializes a string representation to a computation node.
     *
     * @param serialized The string representation of the node.
     * @return The deserialized computation node.
     */
    fun deserializeNode(serialized: String): ComputeNode<T>

    /**
     * Serializes a module to a string representation.
     *
     * @param module The module to serialize.
     * @return A string representation of the module.
     */
    fun serializeModule(module: Module<T>): String

    /**
     * Deserializes a string representation to a module.
     *
     * @param serialized The string representation of the module.
     * @return The deserialized module.
     */
    fun deserializeModule(serialized: String): Module<T>

    /**
     * Saves a computation node to a file.
     *
     * @param node The computation node to save.
     * @param filePath The path to the file where the node will be saved.
     */
    fun saveNode(node: ComputeNode<T>, filePath: String)

    /**
     * Loads a computation node from a file.
     *
     * @param filePath The path to the file containing the serialized node.
     * @return The loaded computation node.
     */
    fun loadNode(filePath: String): ComputeNode<T>

    /**
     * Saves a module to a file.
     *
     * @param module The module to save.
     * @param filePath The path to the file where the module will be saved.
     */
    fun saveModule(module: Module<T>, filePath: String)

    /**
     * Loads a module from a file.
     *
     * @param filePath The path to the file containing the serialized module.
     * @return The loaded module.
     */
    fun loadModule(filePath: String): Module<T>
}