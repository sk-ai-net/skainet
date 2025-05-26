package sk.ai.net.nn

import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor
import kotlin.random.Random

/**
 * Dropout layer that randomly zeroes some of the elements of the input tensor with probability p.
 * 
 * During training, each element is zeroed with probability p, and the remaining elements are scaled by 1/(1-p)
 * to maintain the same expected sum. During evaluation, this module simply returns the input.
 * 
 * @param p Probability of an element to be zeroed. Default: 0.5
 * @param inplace Whether to do the operation in-place. Default: false
 * @param name Name of the module. Default: "Dropout"
 */
class Dropout(
    val p: Double = 0.5,
    val inplace: Boolean = false,
    override val name: String = "Dropout"
) : Module() {
    init {
        require(p in 0.0..1.0) { "Dropout probability has to be between 0 and 1, but got $p" }
    }

    private var training: Boolean = true

    /**
     * Sets the module in training mode.
     */
    fun train() {
        training = true
    }

    /**
     * Sets the module in evaluation mode.
     */
    fun eval() {
        training = false
    }

    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        if (!training || p == 0.0) {
            return input
        }

        if (p == 1.0) {
            // If p is 1, drop everything
            val zeros = DoubleArray(input.size) { 0.0 }
            return DoublesTensor(input.shape, zeros)
        }

        // Create a binary mask with the same shape as the input
        val maskElements = DoubleArray(input.size) { 
            if (Random.nextDouble() < p) 0.0 else 1.0 / (1.0 - p)
        }
        val mask = DoublesTensor(input.shape, maskElements)

        // Apply the mask to the input
        return when (input) {
            is DoublesTensor -> input.times(mask as DoublesTensor)
            else -> {
                // Convert input to DoublesTensor if it's not already
                val inputAsDoubles = input as? DoublesTensor 
                    ?: throw IllegalArgumentException("Input tensor must be a DoublesTensor")
                inputAsDoubles.times(mask as DoublesTensor)
            }
        }
    }
}
