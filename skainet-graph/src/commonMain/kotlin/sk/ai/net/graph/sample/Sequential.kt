package sk.ai.net.graph.sample

import sk.ai.net.graph.nn.Activation
import sk.ai.net.graph.nn.Linear
import sk.ai.net.graph.nn.Sequential
import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.matmul
import sk.ai.net.graph.tensor.plus
import sk.ai.net.graph.tensor.relu

// Define input tensor
val inputTensor = SimpleTensor(listOf(3, 1), doubleArrayOf(0.5, 0.2, -0.3))

fun createSimpleDoubleSequential(input: Double): Sequential<Double> {
    val doubleModel = Sequential<Double>(
        Linear<Double>(
            weights = 2.0,
            bias = 0.5,
            matmul = Double::times,
            add = Double::plus
        ),
        Activation<Double>(::reluDouble)
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
    val weights1 = SimpleTensor(listOf(2, 3), doubleArrayOf(0.1, 0.2, 0.3, -0.4, 0.5, -0.1))
    val bias1 = SimpleTensor(listOf(2, 1), doubleArrayOf(0.1, -0.2))

    val weights2 = SimpleTensor(listOf(1, 2), doubleArrayOf(0.3, -0.5))
    val bias2 = SimpleTensor(listOf(1, 1), doubleArrayOf(0.0))

    // Build your model
    val mlp = Sequential(
        Linear(weights1, bias1, Tensor::matmul, { x, y -> x + y }),
        Activation<Tensor>(::reluTensor),
        Linear(weights2, bias2, Tensor::matmul, { x, y -> x + y })
    )
    return mlp
}



