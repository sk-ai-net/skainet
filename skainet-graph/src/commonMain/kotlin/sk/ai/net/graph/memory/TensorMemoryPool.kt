package sk.ai.net.graph.memory

import sk.ai.net.graph.tensor.shape.Shape

/**
 * A memory pool for tensor data.
 *
 * This class manages a pool of reusable tensor memory, reducing memory allocations
 * and improving performance when working with large tensors. It maintains separate
 * pools for different tensor shapes to avoid fragmentation.
 */
open class TensorMemoryPool {
    /**
     * A map from tensor shapes to pools of available memory blocks.
     */
    private val pools = mutableMapOf<Shape, MutableList<DoubleArray>>()

    /**
     * Acquires a memory block for a tensor with the specified shape.
     *
     * If a suitable memory block is available in the pool, it is reused.
     * Otherwise, a new memory block is allocated.
     *
     * @param shape The shape of the tensor.
     * @return A memory block for the tensor.
     */
    fun acquire(shape: Shape): DoubleArray {
        val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
        val pool = pools.getOrPut(shape) { mutableListOf() }

        return if (pool.isNotEmpty()) {
            pool.removeAt(pool.size - 1)
        } else {
            DoubleArray(size)
        }
    }

    /**
     * Acquires a memory block for a tensor with the specified dimensions.
     * This overload accepts a List<Int> for backward compatibility.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @return A memory block for the tensor.
     */
    fun acquire(dimensions: List<Int>): DoubleArray {
        return acquire(Shape(dimensions.toIntArray()))
    }

    /**
     * Releases a memory block back to the pool.
     *
     * The memory block can be reused for future tensors with the same shape.
     *
     * @param shape The shape of the tensor.
     * @param memory The memory block to release.
     */
    fun release(shape: Shape, memory: DoubleArray) {
        val pool = pools.getOrPut(shape) { mutableListOf() }
        pool.add(memory)
    }

    /**
     * Releases a memory block back to the pool.
     * This overload accepts a List<Int> for backward compatibility.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param memory The memory block to release.
     */
    fun release(dimensions: List<Int>, memory: DoubleArray) {
        release(Shape(dimensions.toIntArray()), memory)
    }

    /**
     * Clears all memory blocks from the pool.
     *
     * This can be used to free up memory when it is no longer needed.
     */
    fun clear() {
        pools.clear()
    }

    /**
     * Gets the number of memory blocks available in the pool for a specific shape.
     *
     * @param shape The shape of the tensor.
     * @return The number of available memory blocks.
     */
    fun getAvailableCount(shape: Shape): Int {
        return pools[shape]?.size ?: 0
    }

    /**
     * Gets the number of memory blocks available in the pool for a specific shape.
     * This overload accepts a List<Int> for backward compatibility.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @return The number of available memory blocks.
     */
    fun getAvailableCount(dimensions: List<Int>): Int {
        return getAvailableCount(Shape(dimensions.toIntArray()))
    }

    /**
     * Gets the total number of memory blocks available in the pool.
     *
     * @return The total number of available memory blocks.
     */
    fun getTotalAvailableCount(): Int {
        return pools.values.sumOf { it.size }
    }

    /**
     * Gets the total amount of memory (in bytes) used by the pool.
     *
     * @return The total amount of memory used by the pool, in bytes.
     */
    fun getTotalMemoryUsage(): Long {
        return pools.entries.sumOf { (shape, pool) ->
            val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
            pool.size * size * Double.SIZE_BYTES.toLong()
        }
    }
}

/**
 * A singleton instance of the tensor memory pool.
 */
object GlobalTensorMemoryPool : TensorMemoryPool()