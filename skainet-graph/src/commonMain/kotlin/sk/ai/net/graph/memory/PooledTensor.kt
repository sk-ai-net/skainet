package sk.ai.net.graph.memory

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A tensor that uses a memory pool to manage its data.
 *
 * This class implements the Tensor interface and uses a TensorMemoryPool to
 * manage its memory. It acquires memory from the pool when created and releases
 * it back to the pool when no longer needed.
 *
 * @property shape The shape of the tensor.
 * @property data The tensor's data, acquired from the memory pool.
 * @property pool The memory pool to use.
 */
class PooledTensor(
    override val shape: Shape,
    private val data: DoubleArray,
    private val pool: TensorMemoryPool = GlobalTensorMemoryPool
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param data The tensor's data, acquired from the memory pool.
     * @param pool The memory pool to use.
     */
    constructor(
        dimensions: List<Int>,
        data: DoubleArray,
        pool: TensorMemoryPool = GlobalTensorMemoryPool
    ) : this(Shape(dimensions.toIntArray()), data, pool)
    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    override fun get(vararg indices: Int): Double {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }
        return data[flatIndex]
    }

    /**
     * Sets the value at the specified indices.
     *
     * @param indices The indices of the element to set.
     * @param value The value to set.
     */
    fun set(vararg indices: Int, value: Double) {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }
        data[flatIndex] = value
    }

    /**
     * Releases the tensor's memory back to the pool.
     *
     * This method should be called when the tensor is no longer needed.
     */
    fun release() {
        pool.release(shape, data)
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape and data.
     */
    override fun toString() =
        "PooledTensor(shape=$shape, data=${data.contentToString()})"

    /**
     * Ensures that the tensor's memory is released when it is garbage collected.
     */
    protected fun finalize() {
        release()
    }

    companion object {
        /**
         * Creates a new pooled tensor with the specified shape.
         *
         * @param shape The shape of the tensor.
         * @param pool The memory pool to use.
         * @return A new pooled tensor.
         */
        fun create(shape: List<Int>, pool: TensorMemoryPool = GlobalTensorMemoryPool): PooledTensor {
            val data = pool.acquire(shape)
            return PooledTensor(shape, data, pool)
        }

        /**
         * Creates a new pooled tensor with the specified shape and initializes it with the given values.
         *
         * @param shape The shape of the tensor.
         * @param values The values to initialize the tensor with.
         * @param pool The memory pool to use.
         * @return A new pooled tensor.
         */
        fun create(shape: List<Int>, values: DoubleArray, pool: TensorMemoryPool = GlobalTensorMemoryPool): PooledTensor {
            val data = pool.acquire(shape)
            values.copyInto(data)
            return PooledTensor(shape, data, pool)
        }

        /**
         * Creates a new pooled tensor with the specified shape and initializes it with zeros.
         *
         * @param shape The shape of the tensor.
         * @param pool The memory pool to use.
         * @return A new pooled tensor.
         */
        fun zeros(shape: List<Int>, pool: TensorMemoryPool = GlobalTensorMemoryPool): PooledTensor {
            val data = pool.acquire(shape)
            data.fill(0.0)
            return PooledTensor(shape, data, pool)
        }

        /**
         * Creates a new pooled tensor with the specified shape and initializes it with ones.
         *
         * @param shape The shape of the tensor.
         * @param pool The memory pool to use.
         * @return A new pooled tensor.
         */
        fun ones(shape: List<Int>, pool: TensorMemoryPool = GlobalTensorMemoryPool): PooledTensor {
            val data = pool.acquire(shape)
            data.fill(1.0)
            return PooledTensor(shape, data, pool)
        }
    }
}
