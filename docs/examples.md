# SK-AI-Net Examples

This document provides examples of common use cases for the SK-AI-Net framework. These examples demonstrate how to use various features of the framework to build and train neural networks, work with tensors, and more.

## Table of Contents

- [Working with Tensors](#working-with-tensors)
- [Building Neural Networks](#building-neural-networks)
- [Automatic Differentiation](#automatic-differentiation)
- [Loading and Saving Models](#loading-and-saving-models)
- [Custom Modules](#custom-modules)

## Working with Tensors

### Creating Tensors

```kotlin
// Create a 2x3 tensor with specific values
val shape = Shape(2, 3)
val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
val tensor = DoublesTensor(shape, data)

// Create a tensor with a single value
val scalarTensor = SimpleTensor(listOf(1), doubleArrayOf(5.0))

// Create a tensor filled with zeros
val zeroTensor = DoublesTensor(Shape(3, 4), DoubleArray(12) { 0.0 })

// Create a tensor filled with ones
val onesTensor = DoublesTensor(Shape(2, 2), DoubleArray(4) { 1.0 })
```

### Accessing Tensor Elements

```kotlin
// Access individual elements
val element = tensor[0, 1] // Returns the element at row 0, column 1

// Get the shape of a tensor
val dimensions = tensor.shape.dimensions // Returns [2, 3]

// Get the total number of elements
val size = tensor.shape.volume // Returns 6
```

### Tensor Operations

```kotlin
// Element-wise addition
val sum = tensor1 + tensor2

// Element-wise multiplication
val product = tensor1 * tensor2

// Matrix multiplication
val matrixProduct = tensor1.matmul(tensor2)

// Element-wise power
val squared = tensor.pow(2.0)

// Apply a function to each element
val activated = tensor.map { if (it > 0.0) it else 0.0 } // ReLU activation
```

### Tensor Expressions

```kotlin
// Create tensors with single values
val a = SimpleTensor(listOf(1), doubleArrayOf(2.0))
val b = SimpleTensor(listOf(1), doubleArrayOf(3.0))
val c = SimpleTensor(listOf(1), doubleArrayOf(4.0))

// Create a tensor expression: (a + b) * c
val expression = (a.expr + b.expr) * c.expr

// Evaluate the expression
val result = expression.node.evaluate()
```

## Building Neural Networks

### Creating a Simple MLP

```kotlin
// Define weights and biases tensors
val weights1 = SimpleTensor(listOf(2, 3), doubleArrayOf(0.1, 0.2, 0.3, -0.4, 0.5, -0.1))
val bias1 = SimpleTensor(listOf(2, 1), doubleArrayOf(0.1, -0.2))

val weights2 = SimpleTensor(listOf(1, 2), doubleArrayOf(0.3, -0.5))
val bias2 = SimpleTensor(listOf(1, 1), doubleArrayOf(0.0))

// Build a multi-layer perceptron
val mlp = Sequential(
    Linear(weights1, bias1, Tensor::matmul, { x, y -> x + y }),
    Activation<Tensor>({ x -> relu(x) }),
    Linear(weights2, bias2, Tensor::matmul, { x, y -> x + y })
)

// Forward pass
val input = SimpleTensor(listOf(3, 1), doubleArrayOf(0.5, 0.2, -0.3))
val output = mlp.forward(input)
```

### Creating a CNN for Image Classification

```kotlin
// Create a simple CNN for image classification
val cnn = Sequential(
    Conv2d(inChannels = 3, outChannels = 16, kernelSize = 3),
    ReLU(),
    MaxPool2d(kernelSize = 2),
    Conv2d(inChannels = 16, outChannels = 32, kernelSize = 3),
    ReLU(),
    MaxPool2d(kernelSize = 2),
    Flatten(),
    Linear(inputSize = 32 * 6 * 6, outputSize = 10)
)

// Forward pass
val inputImage = createImageTensor(/* ... */)
val predictions = cnn.forward(inputImage)
```

## Automatic Differentiation

### Training Mode vs. Inference Mode

```kotlin
// Training mode - gradients are tracked
AutodiffContext.training {
    // Create a tensor that requires gradients
    val x = AutodiffContext.current().tensor(
        Shape(2, 2), 
        doubleArrayOf(1.0, 2.0, 3.0, 4.0), 
        requiresGrad = true
    )
    
    // Perform operations
    val y = x.pow(2.0)
    
    // Compute gradients
    y.backward()
    
    // Access gradients
    val xGrad = (x as AutogradTensor).grad
}

// Inference mode - gradients are not tracked
AutodiffContext.inference {
    val x = AutodiffContext.current().tensor(
        Shape(2, 2), 
        doubleArrayOf(1.0, 2.0, 3.0, 4.0)
    )
    
    // Operations are performed without tracking gradients
    val y = x.pow(2.0)
}
```

### Computing Gradients for Complex Operations

```kotlin
AutodiffContext.training {
    // Create tensors
    val a = AutodiffContext.current().tensor(
        Shape(2, 3), 
        doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), 
        requiresGrad = true
    )
    val b = AutodiffContext.current().tensor(
        Shape(3, 2), 
        doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), 
        requiresGrad = true
    )
    
    // Perform matrix multiplication
    val c = a.matmul(b)
    
    // Create a gradient for c
    val gradC = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 1.0, 1.0, 1.0))
    
    // Compute gradients
    c.backward(gradC)
    
    // Access gradients
    val gradA = (a as AutogradTensor).grad
    val gradB = (b as AutogradTensor).grad
}
```

### Converting Regular Tensors to AutogradTensors

```kotlin
// Create a regular tensor
val regularTensor = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0))

// Convert to AutogradTensor in training mode
AutodiffContext.training {
    // Convert and require gradients
    val autogradTensor = regularTensor.requireGradient()
    
    // Or convert with explicit gradient requirement
    val explicitTensor = regularTensor.withAutodiff(requiresGrad = true)
}
```

## Loading and Saving Models

### Loading from SafeTensors Format

```kotlin
// Load from SafeTensors format
val safeTensorsReader = SafeTensorsReader.fromFilePath("path/to/model.safetensors")
val tensorNames = safeTensorsReader.getTensorNames()

// Create a model
val model = createModel()

// Load tensors into the model
tensorNames.forEach { name ->
    val tensor = safeTensorsReader.readTensor(name)
    // Map the tensor to the appropriate parameter in the model
    mapTensorToModel(model, name, tensor)
}
```

### Using a ParametersLoader

```kotlin
// Create a parameters loader
val loader = CsvParametersLoader { /* source provider */ }

// Load parameters into a model
val model = createModel()
loader.load { name, tensor ->
    // Map the tensor to the appropriate parameter in the model
    mapTensorToModel(model, name, tensor)
}
```

### Using a ModelValuesMapper

```kotlin
// Create a model
val model = createModel()

// Load tensors
val tensors = loadTensors()

// Create a mapper
val mapper = NamesBasedValuesModelMapper()

// Map tensors to model parameters
mapper.mapToModel(model, tensors)
```

## Custom Modules

### Creating a Custom Module

```kotlin
// Define a custom module
class CustomLayer<T>(
    private val transform: (T) -> T
) : Module<T> {
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }
            
            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                return transform(inputValue)
            }
        }
    }
}

// Use the custom module in a model
val model = Sequential(
    Linear(inputSize = 10, outputSize = 5),
    CustomLayer { x -> x.pow(2.0) },
    ReLU()
)
```

### Creating a Custom Activation Function

```kotlin
// Define a custom activation function
fun leakyRelu(x: Tensor, alpha: Double = 0.01): Tensor {
    return x.map { if (it > 0.0) it else alpha * it }
}

// Create an activation module with the custom function
val leakyReluModule = Activation<Tensor> { x -> leakyRelu(x) }

// Use it in a model
val model = Sequential(
    Linear(inputSize = 10, outputSize = 5),
    leakyReluModule
)
```