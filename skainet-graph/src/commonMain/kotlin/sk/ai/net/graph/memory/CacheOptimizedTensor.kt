package sk.ai.net.graph.memory

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A tensor with a memory layout optimized for cache utilization.
 *
 * This class implements the Tensor interface and uses a blocked memory layout
 * to improve cache utilization. It divides the tensor into small blocks that
 * fit in the cache, and stores each block contiguously in memory.
 *
 * @property shape The shape of the tensor.
 * @property blockSize The size of each block in each dimension.
 * @property data The tensor's data, stored in a blocked layout.
 */
class CacheOptimizedTensor(
    override val shape: Shape,
    private val blockSize: List<Int>,
    private val data: DoubleArray
) : Tensor {
    /**
     * Constructor that takes a list of dimensions and converts it to a Shape.
     *
     * @param dimensions The dimensions of the tensor as a list.
     * @param blockSize The size of each block in each dimension.
     * @param data The tensor's data, stored in a blocked layout.
     */
    constructor(
        dimensions: List<Int>,
        blockSize: List<Int>,
        data: DoubleArray
    ) : this(Shape(dimensions.toIntArray()), blockSize, data)
    /**
     * The number of blocks in each dimension.
     */
    private val numBlocks = shape.dimensions.toList().zip(blockSize).map { (dim, block) ->
        (dim + block - 1) / block
    }

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    override fun get(vararg indices: Int): Double {
        val flatIndex = toFlatIndex(indices)
        return data[flatIndex]
    }

    /**
     * Sets the value at the specified indices.
     *
     * @param indices The indices of the element to set.
     * @param value The value to set.
     */
    fun set(vararg indices: Int, value: Double) {
        val flatIndex = toFlatIndex(indices)
        data[flatIndex] = value
    }

    /**
     * Converts multi-dimensional indices to a flat index in the blocked layout.
     *
     * @param indices The indices to convert.
     * @return The flat index in the blocked layout.
     */
    private fun toFlatIndex(indices: IntArray): Int {
        // Calculate the block indices and the indices within the block
        val blockIndices = IntArray(indices.size) { i ->
            indices[i] / blockSize[i]
        }
        val inBlockIndices = IntArray(indices.size) { i ->
            indices[i] % blockSize[i]
        }

        // Calculate the flat index of the block
        var blockFlatIndex = 0
        for (i in blockIndices.indices) {
            var blockStride = 1
            for (j in i + 1 until blockIndices.size) {
                blockStride *= numBlocks[j]
            }
            blockFlatIndex += blockIndices[i] * blockStride
        }

        // Calculate the flat index within the block
        var inBlockFlatIndex = 0
        for (i in inBlockIndices.indices) {
            var inBlockStride = 1
            for (j in i + 1 until inBlockIndices.size) {
                inBlockStride *= blockSize[j]
            }
            inBlockFlatIndex += inBlockIndices[i] * inBlockStride
        }

        // Calculate the size of each block
        val blockElements = blockSize.fold(1, Int::times)

        // Calculate the final flat index
        return blockFlatIndex * blockElements + inBlockFlatIndex
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape and block size.
     */
    override fun toString() =
        "CacheOptimizedTensor(shape=$shape, blockSize=$blockSize)"

    companion object {
        /**
         * Creates a new cache-optimized tensor with the specified shape and block size.
         *
         * @param shape The shape of the tensor.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun create(shape: Shape, blockSize: List<Int>): CacheOptimizedTensor {
            require(shape.dimensions.size == blockSize.size) {
                "Shape and block size must have the same number of dimensions"
            }
            val size = shape.dimensions.fold(1, Int::times)
            val data = DoubleArray(size)
            return CacheOptimizedTensor(shape, blockSize, data)
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun create(dimensions: List<Int>, blockSize: List<Int>): CacheOptimizedTensor {
            return create(Shape(dimensions.toIntArray()), blockSize)
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with zeros.
         *
         * @param shape The shape of the tensor.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun zeros(shape: Shape, blockSize: List<Int>): CacheOptimizedTensor {
            val tensor = create(shape, blockSize)
            tensor.data.fill(0.0)
            return tensor
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with zeros.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun zeros(dimensions: List<Int>, blockSize: List<Int>): CacheOptimizedTensor {
            return zeros(Shape(dimensions.toIntArray()), blockSize)
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with ones.
         *
         * @param shape The shape of the tensor.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun ones(shape: Shape, blockSize: List<Int>): CacheOptimizedTensor {
            val tensor = create(shape, blockSize)
            tensor.data.fill(1.0)
            return tensor
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with ones.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param blockSize The size of each block in each dimension.
         * @return A new cache-optimized tensor.
         */
        fun ones(dimensions: List<Int>, blockSize: List<Int>): CacheOptimizedTensor {
            return ones(Shape(dimensions.toIntArray()), blockSize)
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with the given values.
         *
         * @param shape The shape of the tensor.
         * @param blockSize The size of each block in each dimension.
         * @param values The values to initialize the tensor with.
         * @return A new cache-optimized tensor.
         */
        fun fromValues(shape: Shape, blockSize: List<Int>, values: DoubleArray): CacheOptimizedTensor {
            val tensor = create(shape, blockSize)

            // Copy the values to the tensor in the blocked layout
            val size = shape.dimensions.fold(1, Int::times)
            require(values.size == size) {
                "Values array size (${values.size}) does not match tensor size ($size)"
            }

            // For each element in the tensor
            val indices = IntArray(shape.dimensions.size)
            for (i in 0 until size) {
                // Calculate the multi-dimensional indices
                var remaining = i
                for (j in shape.dimensions.size - 1 downTo 0) {
                    indices[j] = remaining % shape.dimensions[j]
                    remaining /= shape.dimensions[j]
                }

                // Set the value in the blocked layout
                tensor.set(*indices, value = values[i])
            }

            return tensor
        }

        /**
         * Creates a new cache-optimized tensor with the specified shape and block size,
         * and initializes it with the given values.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param blockSize The size of each block in each dimension.
         * @param values The values to initialize the tensor with.
         * @return A new cache-optimized tensor.
         */
        fun fromValues(dimensions: List<Int>, blockSize: List<Int>, values: DoubleArray): CacheOptimizedTensor {
            return fromValues(Shape(dimensions.toIntArray()), blockSize, values)
        }
    }
}
