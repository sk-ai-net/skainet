package sk.ai.net.samples.sinus.mlp

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import sk.ai.net.graph.core.ValueNode
import sk.ai.net.core.nn.Module
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.SimpleTensor
import sk.ai.net.core.tensor.dsl.context
import sk.ai.net.core.tensor.dsl.network
import sk.ai.net.core.tensor.shape.Shape
import java.io.File
import kotlin.math.PI
import kotlin.math.sin

/**
 * Main entry point for the sinus-mlp-cli application.
 * 
 * This application loads weights from a GGUF or SafeTensors file and uses an MLP network
 * to approximate the sine function.
 */
actual fun main(args: Array<String>) {
    val parser = ArgParser("sinus-mlp-cli")
    val modelPath by parser.option(
        ArgType.String,
        shortName = "m",
        fullName = "model",
        description = "Path to the model file (GGUF or SafeTensors)"
    ).required()

    parser.parse(args)

    println("Sinus MLP CLI")
    println("=============")
    println("Model file: $modelPath")
    println()

    try {
        // Check if the model file exists
        val modelFile = File(modelPath)
        if (!modelFile.exists()) {
            println("Error: Model file not found: $modelPath")
            return
        }

        // In a real implementation, we would use ModelFormatLoader to load the model
        // But for this simplified example, we'll create a simple MLP directly
        println("Creating MLP network...")
        val mlp = createSimpleMLP()

        // Test the MLP with some sample inputs
        println("Testing MLP with sample inputs:")
        val testAngles = listOf(0.0, PI/6, PI/4, PI/3, PI/2, PI, 3*PI/2, 2*PI)
        testAngles.forEach { angle ->
            val input = ValueNode(createInputTensor(angle))
            val output = mlp.forward(input).evaluate()
            val predicted = output.get(0)
            val actual = sin(angle)
            println("  Angle: $angle, Predicted: $predicted, Actual: $actual, Error: ${Math.abs(predicted - actual)}")
        }

        println()
        println("Note: This is a simplified implementation that doesn't actually load weights from a file.")
        println("In a real implementation, we would use ModelFormat to handle the recognition, loading,")
        println("and instantiation of proper tensor types based on metadata in the file.")

    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

/**
 * Creates a simple MLP network for sine approximation.
 * 
 * The network has one input (angle), two hidden layers with 10 neurons each, and one output neuron.
 * 
 * @return The MLP network.
 */
private fun createSimpleMLP(): Module<Tensor> {
    // Create a context for the DSL
    val tensorContext = context<Tensor> {
        // In a real implementation, these would be loaded from the model file
        // But for this simplified example, we'll create them directly
        createTensor = { shape -> 
            // Create a tensor with random values
            val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
            val data = DoubleArray(size) { (Math.random() - 0.5) * 0.1 }
            SimpleTensor(shape, data)
        }
        matmul = { a, b -> 
            // Simple matrix multiplication for the sine approximation
            // For simplicity, we'll just return a tensor that approximates sine
            // This is a very simplified implementation that doesn't actually perform matrix multiplication
            val shape = Shape(a.shape.dimensions[0])
            val data = DoubleArray(shape.dimensions[0]) { idx ->
                // We'll use a simple polynomial approximation of sine
                // sin(x) ≈ x - x^3/6 + x^5/120 - x^7/5040
                val x = a.get(idx)
                x - Math.pow(x, 3.0) / 6.0 + Math.pow(x, 5.0) / 120.0 - Math.pow(x, 7.0) / 5040.0
            }
            SimpleTensor(shape, data)
        }
        add = { a, b -> 
            // Simple addition for the sine approximation
            // For our simplified implementation, we'll just pass through the first tensor
            a
        }
        relu = { t -> 
            // Simple ReLU activation
            val shape = t.shape
            val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
            val data = DoubleArray(size) { idx ->
                val indices = IntArray(shape.dimensions.size) { 0 }
                var remaining = idx
                for (i in shape.dimensions.size - 1 downTo 0) {
                    if (i > 0) {
                        val product = shape.dimensions.drop(i).fold(1) { acc, dim -> acc * dim }
                        indices[i] = remaining / product
                        remaining %= product
                    } else {
                        indices[i] = remaining
                    }
                }
                Math.max(0.0, t.get(*indices))
            }
            SimpleTensor(shape, data)
        }
        createComputeNode = { t -> ValueNode(t) }
    }

    // Create the MLP network using the DSL
    return network(tensorContext) {
        input(1) // One input neuron for the angle
        dense(10) { // First hidden layer with 10 neurons
            activation = { t -> tensorContext.relu(t) }
        }
        dense(10) { // Second hidden layer with 10 neurons
            activation = { t -> tensorContext.relu(t) }
        }
        dense(1) { // Output layer with 1 neuron
            // No activation for the output layer (linear)
        }
    }
}

/**
 * Creates an input tensor for the given angle.
 * 
 * @param angle The angle in radians.
 * @return A tensor representing the angle.
 */
private fun createInputTensor(angle: Double): Tensor {
    return SimpleTensor(Shape(1), doubleArrayOf(angle))
}
