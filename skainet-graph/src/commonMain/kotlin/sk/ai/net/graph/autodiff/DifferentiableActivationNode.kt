package sk.ai.net.graph.autodiff

/**
 * A differentiable node that applies an activation function to its input.
 *
 * This node implements [DifferentiableComputeNode], providing
 * automatic differentiation for activation functions.
 *
 * @param T The type of the value to activate.
 * @property activationFunc The activation function to apply.
 * @property gradientFunc The function to compute the gradient of the activation function.
 * @property name The name of the activation function, used for debugging.
 */
class DifferentiableActivationNode<T>(
    private val activationFunc: (T) -> T,
    private val gradientFunc: (T, T) -> T,
    private val name: String = "DifferentiableActivation"
) : DifferentiableComputeNode<T>() {
    /**
     * Evaluates the input node and applies the activation function to its result.
     *
     * @return The result of applying the activation function to the input value.
     */
    override fun evaluate(): T = activationFunc(inputs[0].evaluate())

    /**
     * Computes the gradient of this node with respect to its input.
     *
     * For an activation function f(x), the gradient is f'(x) * gradient,
     * where f'(x) is the derivative of f at x.
     *
     * @param inputIndex The index of the input to compute the gradient with respect to.
     * @param gradient The gradient of the output with respect to this node.
     * @return The gradient of this node with respect to its input.
     */
    override fun gradient(inputIndex: Int, gradient: T): T {
        // For activation functions, we need the derivative of the activation function
        // at the input value, multiplied by the output gradient
        val input = inputs[0].evaluate()
        return gradientFunc(input, gradient)
    }

    /**
     * Returns the name of the activation function.
     *
     * @return The name of the activation function.
     */
    override fun toString(): String = name
}