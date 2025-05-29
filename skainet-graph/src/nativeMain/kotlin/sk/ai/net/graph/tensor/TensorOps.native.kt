package sk.ai.net.graph.tensor


import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import kotlin.math.max

actual operator fun Tensor.plus(other: Tensor): Tensor = NativeTensorOps.add(this, other)
actual operator fun Tensor.times(other: Tensor): Tensor = NativeTensorOps.multiply(this, other)
actual infix fun Tensor.matmul(other: Tensor): Tensor = NativeTensorOps.matmul(this, other)
actual fun relu(tensor: Tensor): Tensor = NativeTensorOps.relu(tensor)


// Simple JVM backend implementation example
object NativeTensorOps {
    fun add(a: Tensor, b: Tensor): Tensor = object : Tensor {
        override val shape = a.shape
        override fun get(vararg indices: Int): Double = a.get(*indices) + b.get(*indices)

    }

    fun multiply(a: Tensor, b: Tensor): Tensor = object : Tensor {
        override val shape = a.shape
        override fun get(vararg indices: Int): Double = a.get(*indices) + b.get(*indices)
    }

    fun matmul(a: Tensor, b: Tensor): Tensor {
        val m = a.shape[0]
        val k = a.shape[1]
        val n = b.shape[1]
        val result = DoubleArray(m * n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var sum = 0.0
                for (x in 0 until k) {
                    sum += a[i, x] * b[x, j]
                }
                result[i * n + j] = sum
            }
        }
        return SimpleTensor(listOf(m, n), result)
    }

    fun relu(tensor: Tensor): Tensor = object : Tensor {
        override val shape = tensor.shape
        override fun get(vararg indices: Int) = max(0.0, tensor.get(*indices))
    }
}
