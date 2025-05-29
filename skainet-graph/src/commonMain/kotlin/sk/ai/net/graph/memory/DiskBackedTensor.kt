package sk.ai.net.graph.memory

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A tensor that stores its data on disk, loading it into memory only when needed.
 *
 * This class implements the Tensor interface and provides out-of-core computation
 * capabilities, allowing tensors to be stored on disk when they don't fit in memory.
 * It uses a memory-mapped file to access the tensor data, which is loaded into memory
 * only when needed.
 *
 * @property shape The shape of the tensor.
 * @property filePath The path to the file where the tensor data is stored.
 */
expect class DiskBackedTensor(
    shape: Shape,
    filePath: String
) : Tensor {
    /**
     * The shape of the tensor.
     */
    override val shape: Shape

    /**
     * The path to the file where the tensor data is stored.
     */
    val filePath: String

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    override fun get(vararg indices: Int): Double

    /**
     * Sets the value at the specified indices.
     *
     * @param indices The indices of the element to set.
     * @param value The value to set.
     */
    fun set(vararg indices: Int, value: Double)

    /**
     * Flushes any changes to disk.
     *
     * This method ensures that any changes made to the tensor are written to disk.
     */
    fun flush()

    /**
     * Closes the tensor, releasing any resources.
     *
     * This method should be called when the tensor is no longer needed.
     */
    fun close()

    companion object {
        /**
         * Creates a new disk-backed tensor with the specified shape.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun create(shape: Shape, filePath: String): DiskBackedTensor

        /**
         * Creates a new disk-backed tensor with the specified shape and initializes it with zeros.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun zeros(shape: Shape, filePath: String): DiskBackedTensor

        /**
         * Creates a new disk-backed tensor with the specified shape and initializes it with ones.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun ones(shape: Shape, filePath: String): DiskBackedTensor

        /**
         * Creates a new disk-backed tensor with the specified dimensions.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun create(dimensions: List<Int>, filePath: String): DiskBackedTensor

        /**
         * Creates a new disk-backed tensor with the specified dimensions and initializes it with zeros.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun zeros(dimensions: List<Int>, filePath: String): DiskBackedTensor

        /**
         * Creates a new disk-backed tensor with the specified dimensions and initializes it with ones.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        fun ones(dimensions: List<Int>, filePath: String): DiskBackedTensor
    }
}
