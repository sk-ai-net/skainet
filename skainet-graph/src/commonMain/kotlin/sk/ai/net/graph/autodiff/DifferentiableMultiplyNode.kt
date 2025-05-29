package sk.ai.net.graph.autodiff

/**
 * A differentiable node that multiplies two inputs.
 *
 * This node implements [DifferentiableComputeNode], providing
 * automatic differentiation for multiplication operations.
 *
 * @param T The type of the values to multiply.
 * @property multiply A function that defines how to multiply two values of type [T].
 */
class DifferentiableMultiplyNode<T>(
    private val multiply: (T, T) -> T
) : DifferentiableComputeNode<T>() {
    /**
     * Evaluates the two input nodes and multiplies their results.
     *
     * @return The result of multiplying the two input values.
     */
    override fun evaluate(): T = multiply(inputs[0].evaluate(), inputs[1].evaluate())

    /**
     * Computes the gradient of this node with respect to the specified input.
     *
     * For multiplication, the gradient depends on the other input:
     * - d(a * b)/da = b
     * - d(a * b)/db = a
     *
     * @param inputIndex The index of the input to compute the gradient with respect to.
     * @param gradient The gradient of the output with respect to this node.
     * @return The gradient of this node with respect to the specified input.
     */
    override fun gradient(inputIndex: Int, gradient: T): T {
        // For multiplication, the gradient is the other input multiplied by the output gradient
        val otherInputIndex = 1 - inputIndex
        val otherInput = inputs[otherInputIndex].evaluate()
        
        // Multiply the gradient by the other input
        return multiply(gradient, otherInput)
    }
}