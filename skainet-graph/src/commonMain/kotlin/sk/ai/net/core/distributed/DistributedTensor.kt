package sk.ai.net.core.distributed

import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.shape.Shape

/**
 * A tensor that is distributed across multiple computation nodes.
 *
 * This class represents a tensor that is partitioned and distributed across
 * multiple computation nodes. It provides methods for accessing and manipulating
 * the distributed tensor, and for performing operations on it.
 *
 * @property shape The shape of the tensor.
 * @property partitions The partitions of the tensor, each assigned to a computation node.
 */
class DistributedTensor(
    override val shape: Shape,
    val partitions: List<TensorPartition>
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param partitions The partitions of the tensor, each assigned to a computation node.
     */
    constructor(
        dimensions: List<Int>,
        partitions: List<TensorPartition>
    ) : this(Shape(dimensions.toIntArray()), partitions)
    /**
     * Retrieves the value at the specified indices.
     *
     * This method finds the partition that contains the specified indices and
     * retrieves the value from that partition.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    override fun get(vararg indices: Int): Double {
        // Find the partition that contains the specified indices
        val partition = findPartition(indices)

        // Convert the global indices to local indices within the partition
        val localIndices = toLocalIndices(indices, partition)

        // Retrieve the value from the partition
        return partition.tensor.get(*localIndices)
    }

    /**
     * Finds the partition that contains the specified indices.
     *
     * @param indices The indices to find the partition for.
     * @return The partition that contains the specified indices.
     * @throws IllegalArgumentException If no partition contains the specified indices.
     */
    private fun findPartition(indices: IntArray): TensorPartition {
        for (partition in partitions) {
            if (partition.containsIndices(indices)) {
                return partition
            }
        }
        throw IllegalArgumentException("No partition contains the specified indices: ${indices.joinToString()}")
    }

    /**
     * Converts global indices to local indices within a partition.
     *
     * @param globalIndices The global indices.
     * @param partition The partition to convert the indices for.
     * @return The local indices within the partition.
     */
    private fun toLocalIndices(globalIndices: IntArray, partition: TensorPartition): IntArray {
        return globalIndices.mapIndexed { i, index ->
            index - partition.startIndices[i]
        }.toIntArray()
    }
}

/**
 * A partition of a distributed tensor.
 *
 * This class represents a partition of a distributed tensor that is assigned to
 * a specific computation node. It contains a subset of the tensor's data and
 * information about the partition's location within the overall tensor.
 *
 * @property tensor The tensor data for this partition.
 * @property startIndices The starting indices of this partition within the overall tensor.
 * @property endIndices The ending indices (exclusive) of this partition within the overall tensor.
 * @property nodeId The ID of the computation node that this partition is assigned to.
 */
class TensorPartition(
    val tensor: Tensor,
    val startIndices: IntArray,
    val endIndices: IntArray,
    val nodeId: String
) {
    /**
     * Checks if this partition contains the specified indices.
     *
     * @param indices The indices to check.
     * @return True if this partition contains the specified indices, false otherwise.
     */
    fun containsIndices(indices: IntArray): Boolean {
        for (i in indices.indices) {
            val index = indices[i]
            if (index < startIndices[i] || index >= endIndices[i]) {
                return false
            }
        }
        return true
    }
}
