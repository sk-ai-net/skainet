package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode

/**
 * A flatten layer.
 *
 * @param T The type of data processed by this module.
 * @param startDim The first dim to flatten (inclusive). Default: 1.
 * @param endDim The last dim to flatten (inclusive). Default: -1.
 * @param name The name of this module.
 * @param flatten The function to perform flattening.
 */
class Flatten<T>(
    private val startDim: Int = 1,
    private val endDim: Int = -1,
    private val name: String = "flatten",
    private val flatten: (T, Int, Int) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply the flattening to the input
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }

            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                return flatten(inputValue, startDim, endDim)
            }
        }
    }
}