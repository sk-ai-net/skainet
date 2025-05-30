package sk.ai.net.core.memory

import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.shape.Shape
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * A JVM implementation of the DiskBackedTensor class.
 *
 * This implementation uses memory-mapped files to store tensor data on disk,
 * loading it into memory only when needed.
 *
 * @property shape The shape of the tensor.
 * @property filePath The path to the file where the tensor data is stored.
 */
actual class DiskBackedTensor actual constructor(
    actual override val shape: Shape,
    actual val filePath: String
) : Tensor, AutoCloseable {
    /**
     * The size of the tensor (total number of elements).
     */
    private val size = shape.volume

    /**
     * The file channel for the memory-mapped file.
     */
    private val fileChannel: FileChannel

    /**
     * The memory-mapped buffer for accessing the tensor data.
     */
    private val buffer: MappedByteBuffer

    init {
        // Create the file and memory-mapped buffer
        val file = RandomAccessFile(filePath, "rw")
        fileChannel = file.channel
        buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, size * Double.SIZE_BYTES.toLong())
    }

    /**
     * Retrieves the value at the specified indices.
     *
     * @param indices The indices of the element to retrieve.
     * @return The value at the specified indices.
     */
    actual override fun get(vararg indices: Int): Double {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }
        return buffer.getDouble(flatIndex * Double.SIZE_BYTES)
    }

    /**
     * Sets the value at the specified indices.
     *
     * @param indices The indices of the element to set.
     * @param value The value to set.
     */
    actual fun set(vararg indices: Int, value: Double) {
        val flatIndex = indices.foldIndexed(0) { index, acc, i ->
            acc * shape.dimensions[index] + i
        }
        buffer.putDouble(flatIndex * Double.SIZE_BYTES, value)
    }

    /**
     * Flushes any changes to disk.
     *
     * This method ensures that any changes made to the tensor are written to disk.
     */
    actual fun flush() {
        buffer.force()
    }

    /**
     * Closes the tensor, releasing any resources.
     *
     * This method should be called when the tensor is no longer needed.
     */
    actual override fun close() {
        fileChannel.close()
    }

    /**
     * Returns a string representation of the tensor.
     *
     * @return A string representation of the tensor, including its shape and file path.
     */
    override fun toString() =
        "DiskBackedTensor(shape=$shape, filePath=$filePath)"

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
            val tensor = DiskBackedTensor(shape, filePath)

            // Initialize the tensor with zeros
            for (i in 0 until tensor.size) {
                tensor.buffer.putDouble(i * Double.SIZE_BYTES, 0.0)
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
            val tensor = DiskBackedTensor(shape, filePath)

            // Initialize the tensor with ones
            for (i in 0 until tensor.size) {
                tensor.buffer.putDouble(i * Double.SIZE_BYTES, 1.0)
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
