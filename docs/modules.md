# Module System and Custom Modules in SK-AI-Net

This document provides a comprehensive guide to the module system in SK-AI-Net and explains how to create custom modules for your specific needs.

## Table of Contents

- [Module System Overview](#module-system-overview)
- [Built-in Modules](#built-in-modules)
- [Creating Custom Modules](#creating-custom-modules)
- [Module Composition](#module-composition)
- [Parameter Management](#parameter-management)
- [Advanced Module Techniques](#advanced-module-techniques)

## Module System Overview

The module system in SK-AI-Net is designed to be flexible, extensible, and type-safe. It provides a way to encapsulate neural network layers and operations into reusable components.

### Core Module Interface

At the heart of the module system is the `Module` interface:

```kotlin
interface Module<T> {
    fun forward(input: ComputeNode<T>): ComputeNode<T>
    
    // Optional list of child modules
    val modules: List<Module<T>> get() = emptyList()
}
```

This simple interface allows for:
- Type-parameterized modules that can work with different data types
- Composition of modules into more complex structures
- A consistent API for forward propagation

## Built-in Modules

SK-AI-Net provides several built-in modules that cover common neural network layers and operations:

### Linear Layers

```kotlin
// Basic linear layer (fully connected)
val linear = Linear<Tensor>(
    inputSize = 784,
    outputSize = 128
)

// Linear layer with custom initialization
val linearCustom = Linear<Tensor>(
    weights = initializedWeights,
    bias = initializedBias,
    matmul = Tensor::matmul,
    add = { x, y -> x + y }
)
```

### Activation Functions

```kotlin
// ReLU activation
val relu = ReLU<Tensor>()

// Custom activation
val leakyRelu = Activation<Tensor> { x -> 
    x.map { if (it > 0.0) it else 0.01 * it }
}
```

### Convolutional Layers

```kotlin
// 2D Convolution
val conv = Conv2d<Tensor>(
    inChannels = 3,
    outChannels = 16,
    kernelSize = 3,
    stride = 1,
    padding = 1
)
```

### Pooling Layers

```kotlin
// Max pooling
val maxPool = MaxPool2d<Tensor>(
    kernelSize = 2,
    stride = 2
)
```

### Container Modules

```kotlin
// Sequential container
val model = Sequential(
    Linear<Tensor>(inputSize = 784, outputSize = 128),
    ReLU<Tensor>(),
    Linear<Tensor>(inputSize = 128, outputSize = 10)
)
```

## Creating Custom Modules

Creating custom modules in SK-AI-Net is straightforward. You can either implement the `Module` interface directly or extend existing modules.

### Implementing the Module Interface

Here's a simple example of a custom module that applies a custom transformation to its input:

```kotlin
class CustomTransform<T>(
    private val transform: (T) -> T,
    private val name: String = "CustomTransform"
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
            
            override fun toString(): String = name
        }
    }
}
```

Usage:
```kotlin
// Create a custom module that squares each element
val squareModule = CustomTransform<Tensor> { tensor ->
    tensor.map { it * it }
}

// Use it in a model
val model = Sequential(
    Linear<Tensor>(inputSize = 10, outputSize = 5),
    squareModule,
    ReLU<Tensor>()
)
```

### Extending Existing Modules

You can also extend existing modules to customize their behavior:

```kotlin
class LeakyReLU<T>(
    private val alpha: Double = 0.01,
    private val leakyRelu: (T, Double) -> T
) : Activation<T>({ x -> leakyRelu(x, alpha) }, "LeakyReLU")
```

Usage:
```kotlin
// Create a leaky ReLU module for tensors
val leakyRelu = LeakyReLU<Tensor> { tensor, alpha ->
    tensor.map { if (it > 0.0) it else alpha * it }
}
```

## Module Composition

Modules can be composed to create more complex architectures:

### Sequential Composition

The `Sequential` module allows for linear chaining of modules:

```kotlin
val model = Sequential(
    Linear<Tensor>(inputSize = 784, outputSize = 128),
    ReLU<Tensor>(),
    Linear<Tensor>(inputSize = 128, outputSize = 10)
)
```

### Parallel Composition

You can create custom modules that process inputs in parallel and combine the results:

```kotlin
class Parallel<T>(
    private val branches: List<Module<T>>,
    private val combiner: (List<T>) -> T
) : Module<T> {
    override val modules: List<Module<T>> get() = branches
    
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        return object : ComputeNode<T>() {
            private val branchOutputs = branches.map { it.forward(input) }
            
            init {
                inputs.addAll(branchOutputs)
            }
            
            override fun evaluate(): T {
                val results = inputs.map { it.evaluate() }
                return combiner(results)
            }
        }
    }
}
```

Usage:
```kotlin
// Create a parallel module with two branches
val parallel = Parallel<Tensor>(
    branches = listOf(
        Linear<Tensor>(inputSize = 10, outputSize = 5),
        Linear<Tensor>(inputSize = 10, outputSize = 5)
    ),
    combiner = { results -> results[0] + results[1] } // Sum the outputs
)
```

### Residual Connections

You can implement residual connections (skip connections) using custom modules:

```kotlin
class ResidualBlock<T>(
    private val module: Module<T>,
    private val add: (T, T) -> T
) : Module<T> {
    override val modules: List<Module<T>> get() = listOf(module)
    
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        val moduleOutput = module.forward(input)
        
        return object : ComputeNode<T>() {
            init {
                inputs += input
                inputs += moduleOutput
            }
            
            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                val moduleValue = inputs[1].evaluate()
                return add(inputValue, moduleValue)
            }
        }
    }
}
```

Usage:
```kotlin
// Create a residual block
val residual = ResidualBlock(
    module = Sequential(
        Linear<Tensor>(inputSize = 128, outputSize = 128),
        ReLU<Tensor>(),
        Linear<Tensor>(inputSize = 128, outputSize = 128)
    ),
    add = { x, y -> x + y }
)
```

## Parameter Management

For modules that need to manage parameters (weights, biases, etc.), SK-AI-Net provides the `ModuleParameters` interface:

```kotlin
interface ModuleParameters {
    val name: String
    val params: List<Parameter<*>>
}

data class Parameter<T>(
    val name: String,
    var value: T
)
```

Implementing this interface allows your module to participate in parameter initialization, optimization, and serialization:

```kotlin
class CustomLinear<T>(
    private val inputSize: Int,
    private val outputSize: Int,
    initialWeights: T? = null,
    initialBias: T? = null,
    private val matmul: (T, T) -> T,
    private val add: (T, T) -> T,
    override val name: String = "linear"
) : Module<T>, ModuleParameters {
    // Parameters
    private val weightsParam = Parameter("weights", 
        initialWeights ?: createRandomTensor(outputSize, inputSize))
    private val biasParam = Parameter("bias", 
        initialBias ?: createZerosTensor(outputSize))
    
    override val params: List<Parameter<*>> = listOf(weightsParam, biasParam)
    
    @Suppress("UNCHECKED_CAST")
    private val weights: T get() = weightsParam.value as T
    
    @Suppress("UNCHECKED_CAST")
    private val bias: T get() = biasParam.value as T
    
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }
            
            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                val output = matmul(inputValue, weights)
                return add(output, bias)
            }
        }
    }
}
```

## Advanced Module Techniques

### Stateful Modules

Some modules need to maintain state between forward passes. You can implement stateful modules by storing state in member variables:

```kotlin
class BatchNorm<T>(
    private val numFeatures: Int,
    private val epsilon: Double = 1e-5,
    private val momentum: Double = 0.1,
    private val createTensor: (Shape, DoubleArray) -> T,
    private val mean: (T) -> T,
    private val variance: (T) -> T,
    private val normalize: (T, T, T, Double) -> T
) : Module<T>, ModuleParameters {
    // Parameters
    private val gammaParam = Parameter("gamma", 
        createTensor(Shape(numFeatures), DoubleArray(numFeatures) { 1.0 }))
    private val betaParam = Parameter("beta", 
        createTensor(Shape(numFeatures), DoubleArray(numFeatures) { 0.0 }))
    
    // Running statistics (state)
    private var runningMean: T = createTensor(Shape(numFeatures), DoubleArray(numFeatures) { 0.0 })
    private var runningVar: T = createTensor(Shape(numFeatures), DoubleArray(numFeatures) { 1.0 })
    
    override val name: String = "batch_norm"
    override val params: List<Parameter<*>> = listOf(gammaParam, betaParam)
    
    @Suppress("UNCHECKED_CAST")
    private val gamma: T get() = gammaParam.value as T
    
    @Suppress("UNCHECKED_CAST")
    private val beta: T get() = betaParam.value as T
    
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        // Implementation details omitted for brevity
    }
}
```

### Conditional Modules

You can create modules that behave differently based on conditions:

```kotlin
class Dropout<T>(
    private val rate: Double,
    private val scale: (T, Double) -> T,
    private val multiply: (T, T) -> T,
    private val createDropoutMask: (Shape, Double) -> T
) : Module<T> {
    private var isTraining: Boolean = true
    
    fun train() { isTraining = true }
    fun eval() { isTraining = false }
    
    override fun forward(input: ComputeNode<T>): ComputeNode<T> {
        return object : ComputeNode<T>() {
            init {
                inputs += input
            }
            
            override fun evaluate(): T {
                val inputValue = inputs[0].evaluate()
                
                if (!isTraining) {
                    return inputValue // No dropout during evaluation
                }
                
                // Create dropout mask and apply it
                val shape = getShape(inputValue)
                val mask = createDropoutMask(shape, rate)
                val scaled = scale(inputValue, 1.0 / (1.0 - rate))
                return multiply(scaled, mask)
            }
        }
    }
}
```

By understanding and utilizing these module system features, you can create custom neural network architectures tailored to your specific needs while leveraging the performance and flexibility of the SK-AI-Net framework.