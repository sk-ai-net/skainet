package sk.ai.net.graph.backend

import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.ShapeChecker
import kotlin.math.max

/**
 * Native-specific implementation of the CPU backend.
 *
 * This backend provides optimized implementations of tensor operations for the Native platform.
 */
class NativeCpuBackend : CpuBackend() {
    /**
     * The name of the backend.
     */
    override val name: String = "Native CPU"

    /**
     * Adds two tensors element-wise.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of adding the two tensors.
     * @throws IllegalArgumentException If the tensor shapes are not compatible.
     */
    override fun add(left: Tensor, right: Tensor): Tensor {
        // Check that the shapes are compatible
        val resultShape = ShapeChecker.computeElementWiseShape(left, right, "addition")

        return object : Tensor {
            override val shape = resultShape
            override fun get(vararg indices: Int): Double = left.get(*indices) + right.get(*indices)
        }
    }

    /**
     * Multiplies two tensors element-wise.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of multiplying the two tensors.
     * @throws IllegalArgumentException If the tensor shapes are not compatible.
     */
    override fun multiply(left: Tensor, right: Tensor): Tensor {
        // Check that the shapes are compatible
        val resultShape = ShapeChecker.computeElementWiseShape(left, right, "multiplication")

        return object : Tensor {
            override val shape = resultShape
            override fun get(vararg indices: Int): Double = left.get(*indices) * right.get(*indices)
        }
    }

    /**
     * Performs matrix multiplication of two tensors.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The result of matrix multiplication.
     * @throws IllegalArgumentException If the tensor shapes are not compatible.
     */
    override fun matmul(left: Tensor, right: Tensor): Tensor {
        // Check that the shapes are compatible
        val resultShape = ShapeChecker.computeMatmulShape(left, right)

        val m = resultShape[0]
        val n = resultShape[1]
        val k = left.shape[1]
        val result = DoubleArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0.0
                for (x in 0 until k) {
                    sum += left[i, x] * right[x, j]
                }
                result[i * n + j] = sum
            }
        }
        return SimpleTensor(resultShape, result)
    }

    /**
     * Applies the ReLU activation function to a tensor.
     *
     * @param tensor The input tensor.
     * @return The result of applying ReLU.
     */
    override fun relu(tensor: Tensor): Tensor = object : Tensor {
        override val shape = tensor.shape
        override fun get(vararg indices: Int) = max(0.0, tensor.get(*indices))
    }
}