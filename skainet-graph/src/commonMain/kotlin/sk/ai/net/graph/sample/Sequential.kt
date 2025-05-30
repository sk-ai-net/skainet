package sk.ai.net.graph.sample

import sk.ai.net.core.nn.Activation
import sk.ai.net.core.nn.Linear
import sk.ai.net.core.nn.Sequential
import sk.ai.net.core.tensor.SimpleTensor
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.matmul
import sk.ai.net.core.tensor.plus
import sk.ai.net.core.tensor.relu
import sk.ai.net.core.tensor.shape.Shape

// Define input tensor
val inputTensor = SimpleTensor(Shape(3, 1), doubleArrayOf(0.5, 0.2, -0.3))

fun createSimpleDoubleSequential(input: Double): Sequential<Double> {
    val doubleModel = Sequential<Double>(
        Linear<Double>(
            weights = 2.0,
            bias = 0.5,
            matmul = Double::times,
            add = Double::plus
        ),
        Activation<Double>(::reluDouble, "ss"), name = "name"
    )
    return doubleModel
}

// Simple relu for Double
fun reluDouble(x: Double) = if (x > 0.0) x else 0.0

// Simple relu for Double
fun reluTensor(x: Tensor) = relu(x)


fun createSimpleTensorMLP(): Sequential<Tensor> {
    // This function is just a placeholder to demonstrate where the model would be built.
    // The actual model building code is below.

// Define weights and biases tensors
    val weights1 = SimpleTensor(Shape(2, 3), doubleArrayOf(0.1, 0.2, 0.3, -0.4, 0.5, -0.1))
    val bias1 = SimpleTensor(Shape(2, 1), doubleArrayOf(0.1, -0.2))

    val weights2 = SimpleTensor(Shape(1, 2), doubleArrayOf(0.3, -0.5))
    val bias2 = SimpleTensor(Shape(1, 1), doubleArrayOf(0.0))

    // Build your model
    val mlp = Sequential(
        Linear(weights1, bias1, Tensor::matmul, { x, y -> x + y }),
        Activation<Tensor>(::reluTensor),
        Linear(weights2, bias2, Tensor::matmul, { x, y -> x + y })
    )
    return mlp
}



