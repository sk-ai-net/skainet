package sk.ai.net.core.tensor.dsl

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.core.tensor.shape.Shape

/**
 * Context for the DSL to define the data type and operations.
 *
 * This class holds the information about the data type and operations
 * that should be used in the DSL. It's used to make the DSL generic
 * and to avoid hardcoding the data type.
 *
 * @param T The type of data processed by the modules.
 */
class Context<T> {
    /**
     * Function to create a tensor with the given shape.
     */
    var createTensor: (Shape) -> T = { throw IllegalStateException("createTensor not initialized") }

    /**
     * Function to perform matrix multiplication.
     */
    var matmul: (T, T) -> T = { _, _ -> throw IllegalStateException("matmul not initialized") }

    /**
     * Function to add two tensors.
     */
    var add: (T, T) -> T = { _, _ -> throw IllegalStateException("add not initialized") }

    /**
     * Function to apply the ReLU activation function.
     */
    var relu: (T) -> T = { throw IllegalStateException("relu not initialized") }

    /**
     * Function to create a compute node from a tensor.
     */
    var createComputeNode: (T) -> ComputeNode<T> = { throw IllegalStateException("createComputeNode not initialized") }

    /**
     * Function to perform convolution.
     */
    var convolution: (T, Shape) -> T = { _, _ -> throw IllegalStateException("convolution not initialized") }

    /**
     * Function to perform max pooling.
     */
    var maxPool: (T, Shape) -> T = { _, _ -> throw IllegalStateException("maxPool not initialized") }

    /**
     * Function to perform dropout.
     */
    var dropout: (T, Double, Boolean) -> T = { t, _, _ -> throw IllegalStateException("dropout not initialized") }

    /**
     * Function to perform flattening.
     */
    var flatten: (T, Int, Int) -> T = { t, _, _ -> throw IllegalStateException("flatten not initialized") }
}

/**
 * Creates a context for the DSL with the given configuration.
 *
 * @param T The type of data processed by the modules.
 * @param init The configuration function.
 * @return The configured context.
 */
fun <T> context(init: Context<T>.() -> Unit): Context<T> {
    val context = Context<T>()
    context.init()
    return context
}
