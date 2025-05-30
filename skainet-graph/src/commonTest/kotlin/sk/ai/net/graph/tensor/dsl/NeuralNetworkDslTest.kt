package sk.ai.net.graph.tensor.dsl

import sk.ai.net.graph.core.ValueNode
import sk.ai.net.core.tensor.Tensor
import sk.ai.net.core.tensor.dsl.Context
import sk.ai.net.core.tensor.dsl.context
import sk.ai.net.core.tensor.dsl.network
import sk.ai.net.core.tensor.shape.Shape
import kotlin.test.Test
import kotlin.test.assertEquals

class NeuralNetworkDslTest {

    // Simple tensor implementation for testing
    class TestTensor(override val shape: Shape, private val data: DoubleArray) : Tensor {
        override fun get(vararg indices: Int): Double {
            val flatIndex = indices.foldIndexed(0) { index, acc, i ->
                acc * shape.dimensions[index] + i
            }
            return data[flatIndex]
        }
    }

    // Create a context for testing
    private fun createTestContext(): Context<Tensor> {
        return context {
            createTensor = { shape ->
                val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
                val data = DoubleArray(size) { 0.1 * it }
                TestTensor(shape, data)
            }

            matmul = { a, b ->
                // Simple matrix multiplication for testing
                val aShape = a.shape.dimensions
                val bShape = b.shape.dimensions

                // Handle different tensor shapes
                val resultShape = when {
                    aShape.size > 1 && bShape.size > 1 -> Shape(aShape[0], bShape[1])
                    aShape.size == 1 && bShape.size == 1 -> Shape(1)
                    aShape.size > 1 -> Shape(aShape[0])
                    else -> Shape(bShape[0])
                }

                val size = resultShape.dimensions.fold(1) { acc, dim -> acc * dim }
                val resultData = DoubleArray(size) { 0.5 }
                TestTensor(resultShape, resultData)
            }

            add = { a, b ->
                // Simple addition for testing
                val shape = a.shape
                val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
                val data = DoubleArray(size) { 0.3 }
                TestTensor(shape, data)
            }

            relu = { tensor ->
                // Simple ReLU for testing
                val shape = tensor.shape
                val size = shape.dimensions.fold(1) { acc, dim -> acc * dim }
                val data = DoubleArray(size) { 0.2 }
                TestTensor(shape, data)
            }

            createComputeNode = { tensor ->
                ValueNode(tensor)
            }

            // Dummy implementations for other required functions
            convolution = { tensor, _ ->
                tensor
            }

            maxPool = { tensor, _ ->
                tensor
            }

            dropout = { tensor, _, _ ->
                tensor
            }

            flatten = { tensor, _, _ ->
                tensor
            }
        }
    }

    @Test
    fun `test MLP with 2 hidden layers for prediction`() {
        val context = createTestContext()

        // Create an MLP with 2 hidden layers for regression
        val mlp = network(context) {
            // Input layer with 4 features
            input(4)

            // First hidden layer with 8 neurons and ReLU activation
            dense(8) {
                activation = context.relu
            }

            // Second hidden layer with 4 neurons and ReLU activation
            dense(4) {
                activation = context.relu
            }

            // Output layer with 1 neuron (for regression)
            dense(1) {
                // Identity activation for regression
                activation = { it }
            }
        }

        // Create input tensor
        val inputTensor = context.createTensor(Shape(4))
        val inputNode = context.createComputeNode(inputTensor)

        // Forward pass
        val outputNode = mlp.forward(inputNode)
        val output = outputNode.evaluate()

        // Check output shape
        assertEquals(1, output.shape.dimensions.size)
        assertEquals(1, output.shape.dimensions[0])
    }

    @Test
    fun `test MLP with 2 hidden layers for classification`() {
        val context = createTestContext()

        // Create an MLP with 2 hidden layers for classification
        val mlp = network(context) {
            // Input layer with 4 features
            input(4)

            // First hidden layer with 8 neurons and ReLU activation
            dense(8) {
                activation = context.relu
            }

            // Second hidden layer with 4 neurons and ReLU activation
            dense(4) {
                activation = context.relu
            }

            // Output layer with 3 neurons (for 3-class classification)
            dense(3) {
                // Identity activation for raw logits
                activation = { it }
            }
        }

        // Create input tensor
        val inputTensor = context.createTensor(Shape(4))
        val inputNode = context.createComputeNode(inputTensor)

        // Forward pass
        val outputNode = mlp.forward(inputNode)
        val output = outputNode.evaluate()

        // Check output shape
        assertEquals(1, output.shape.dimensions.size)
        assertEquals(3, output.shape.dimensions[0])
    }
}
