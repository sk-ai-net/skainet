package sk.ai.net.graph.memory

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Android implementation of DiskBackedTensor.
 *
 * This implementation uses memory-mapped files to store tensor data on disk.
 */
actual class DiskBackedTensor actual constructor(
    shape: Shape,
    filePath: String
) : Tensor {
    /**
     * The shape of the tensor.
     */
    actual override val shape: Shape = shape

    /**
     * The path to the file where the tensor data is stored.
     */
    actual val filePath: String = filePath

    private val file: RandomAccessFile
    private val buffer: MappedByteBuffer
    private val size: Int

    init {
        // Calculate the total size of the tensor
        size = shape.volume

        // Create the file and memory-mapped buffer
        val file = File(filePath)
        file.parentFile?.mkdirs()

        this.file = RandomAccessFile(file, "rw")
        this.buffer = this.file.channel.map(
            FileChannel.MapMode.READ_WRITE,
            0,
            size.toLong() * Double.SIZE_BYTES
        )
        this.buffer.order(ByteOrder.nativeOrder())
    }

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    actual override fun get(vararg indices: Int): Double {
        val index = calculateIndex(indices)
        return buffer.getDouble(index * Double.SIZE_BYTES)
    }

    /**
     * Sets the value at the specified indices.
     *
     * @param indices The indices of the element to set.
     * @param value The value to set.
     */
    actual fun set(vararg indices: Int, value: Double) {
        val index = calculateIndex(indices)
        buffer.putDouble(index * Double.SIZE_BYTES, value)
    }

    /**
     * Flushes any changes to disk.
     */
    actual fun flush() {
        buffer.force()
    }

    /**
     * Closes the tensor, releasing any resources.
     */
    actual fun close() {
        file.close()
    }

    /**
     * Calculates the linear index from the multi-dimensional indices.
     *
     * @param indices The multi-dimensional indices.
     * @return The linear index.
     */
    private fun calculateIndex(indices: IntArray): Int {
        require(indices.size == shape.rank) { "Number of indices must match tensor dimensions" }

        var index = 0
        var stride = 1

        for (i in shape.rank - 1 downTo 0) {
            require(indices[i] >= 0 && indices[i] < shape.dimensions[i]) { "Index out of bounds" }
            index += indices[i] * stride
            stride *= shape.dimensions[i]
        }

        return index
    }

    actual companion object {
        /**
         * Creates a new disk-backed tensor with the specified shape.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun create(shape: Shape, filePath: String): DiskBackedTensor {
            return DiskBackedTensor(shape, filePath)
        }

        /**
         * Creates a new disk-backed tensor with the specified shape and initializes it with zeros.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun zeros(shape: Shape, filePath: String): DiskBackedTensor {
            val tensor = create(shape, filePath)

            for (i in 0 until tensor.size) {
                val buffer = tensor.buffer
                buffer.position(i * Double.SIZE_BYTES)
                buffer.putDouble(0.0)
            }

            tensor.flush()
            return tensor
        }

        /**
         * Creates a new disk-backed tensor with the specified shape and initializes it with ones.
         *
         * @param shape The shape of the tensor.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun ones(shape: Shape, filePath: String): DiskBackedTensor {
            val tensor = create(shape, filePath)

            for (i in 0 until tensor.size) {
                val buffer = tensor.buffer
                buffer.position(i * Double.SIZE_BYTES)
                buffer.putDouble(1.0)
            }

            tensor.flush()
            return tensor
        }

        /**
         * Creates a new disk-backed tensor with the specified dimensions.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun create(dimensions: List<Int>, filePath: String): DiskBackedTensor {
            return create(Shape(dimensions.toIntArray()), filePath)
        }

        /**
         * Creates a new disk-backed tensor with the specified dimensions and initializes it with zeros.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun zeros(dimensions: List<Int>, filePath: String): DiskBackedTensor {
            return zeros(Shape(dimensions.toIntArray()), filePath)
        }

        /**
         * Creates a new disk-backed tensor with the specified dimensions and initializes it with ones.
         * This overload accepts a List<Int> for backward compatibility.
         *
         * @param dimensions The dimensions of the tensor as a list.
         * @param filePath The path to the file where the tensor data will be stored.
         * @return A new disk-backed tensor.
         */
        actual fun ones(dimensions: List<Int>, filePath: String): DiskBackedTensor {
            return ones(Shape(dimensions.toIntArray()), filePath)
        }
    }
}
