package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A 2D max pooling layer.
 *
 * @param T The type of data processed by this module.
 * @param kernelSize Size of the window to take a max over.
 * @param stride Stride of the window. Default value is kernelSize.
 * @param name The name of this module.
 * @param maxPool The function to perform max pooling.
 */
class MaxPool2d<T>(
    private val kernelSize: Int,
    private val stride: Int = kernelSize,
    private val name: String = "maxPool2d",
    private val maxPool: (T, Shape) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply the max pooling to the input
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }

            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                // Create a shape for the max pooling parameters
                val shape = Shape(kernelSize, stride)
                return maxPool(inputValue, shape)
            }
        }
    }
}