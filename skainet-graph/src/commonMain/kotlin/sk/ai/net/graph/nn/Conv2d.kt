package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.graph.tensor.shape.Shape

/**
 * A 2D convolution layer.
 *
 * @param T The type of data processed by this module.
 * @param inChannels Number of channels in the input image.
 * @param outChannels Number of channels produced by the convolution.
 * @param kernelSize Size of the convolving kernel.
 * @param stride Stride of the convolution.
 * @param padding Zero-padding added to both sides of the input.
 * @param name The name of this module.
 * @param convolution The function to perform convolution.
 */
class Conv2d<T>(
    private val inChannels: Int,
    private val outChannels: Int,
    private val kernelSize: Int,
    private val stride: Int = 1,
    private val padding: Int = 0,
    private val name: String = "conv2d",
    private val convolution: (T, Shape) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply the convolution to the input
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }

            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                // Create a shape for the convolution parameters
                val shape = Shape(inChannels, outChannels, kernelSize, stride, padding)
                return convolution(inputValue, shape)
            }
        }
    }
}