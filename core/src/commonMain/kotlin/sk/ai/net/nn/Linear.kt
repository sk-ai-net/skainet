package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.nn.reflection.ModuleParameter
import sk.ai.net.nn.reflection.ModuleParameters
import sk.ai.net.nn.reflection.bias
import sk.ai.net.nn.reflection.weights

/**
 * Linear layer (a.k.a. fully connected dense layer). This layer applies a linear transformation to the input data.
 * The weights and biases are learned during training.
 *
 * @param inFeatures Number of input features
 * @param outFeatures Number of output features
 * @param name Name of the module
 * @param initWeights Initial weights
 * @param initBias Initial bias
 */

class Linear(
    inFeatures: Int,
    outFeatures: Int,
    override val name: String = "Linear",
    initWeights: Tensor = DoublesTensor(
        Shape(outFeatures, inFeatures),
        List(inFeatures * outFeatures) { 0.0 }.map { it }.toDoubleArray()
    ),
    initBias: Tensor = DoublesTensor(
        Shape(outFeatures),
        List(outFeatures) { 0.0 }.map { it }.toDoubleArray()
    ),
    override val params: List<ModuleParameter> = listOf(
        ModuleParameter.WeightParameter("$name.weight", initWeights),
        ModuleParameter.BiasParameter("$name.bias", initBias)
    ),
) : Module(), ModuleParameters {

    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor {
        val weight = params.weights().value
        val bias = params.bias().value

        // matrix multiplication on tensors and addition
        return input.matmul(weight.t()) + bias
    }
}
