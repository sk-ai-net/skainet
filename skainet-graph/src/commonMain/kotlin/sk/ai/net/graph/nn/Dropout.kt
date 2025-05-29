package sk.ai.net.graph.nn

import sk.ai.net.graph.core.ComputeNode

/**
 * A dropout layer.
 *
 * @param T The type of data processed by this module.
 * @param p Probability of an element to be zeroed. Default: 0.5.
 * @param inplace If set to true, will do this operation in-place. Default: false.
 * @param name The name of this module.
 * @param dropout The function to perform dropout.
 */
class Dropout<T>(
    private val p: Double = 0.5,
    private val inplace: Boolean = false,
    private val name: String = "dropout",
    private val dropout: (T, Double, Boolean) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Apply the dropout to the input
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }

            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                return dropout(inputValue, p, inplace)
            }
        }
    }
}