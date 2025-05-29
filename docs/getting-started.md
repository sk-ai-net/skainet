# Getting Started with SK-AI-Net

This guide will help you get started with SK-AI-Net, an open-source deep learning framework written in Kotlin. It covers installation, basic usage, and common examples to help you begin building AI-powered applications.

## Table of Contents

- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Creating Your First Tensor](#creating-your-first-tensor)
- [Building a Simple Neural Network](#building-a-simple-neural-network)
- [Loading Pre-trained Models](#loading-pre-trained-models)
- [Common Use Cases](#common-use-cases)
- [Next Steps](#next-steps)

## Installation

### Gradle

Add the following to your project's `build.gradle.kts`:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/sk-ai-net/skainet")
        credentials {
            username = providers.gradleProperty("gpr.user")
                .orElse(System.getenv("GITHUB_ACTOR"))
                .get()
            password = providers.gradleProperty("gpr.token")
                .orElse(System.getenv("GITHUB_TOKEN"))
                .get()
        }
    }
}

dependencies {
    implementation("sk.ai.net:skainet-graph:0.1.0")
    // Add other modules as needed:
    // implementation("sk.ai.net:safetensors:0.1.0")
    // implementation("sk.ai.net:gguf:0.1.0")
    // implementation("sk.ai.net:io:0.1.0")
}
```

Ensure you provide your GitHub username (`gpr.user`) and a personal access token (`gpr.token`) with package read permission.

### Maven

For Maven projects, add the following to your `pom.xml`:

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/sk-ai-net/skainet</url>
        <snapshots>
            <enabled>true</enabled>
        </snapshots>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>sk.ai.net</groupId>
        <artifactId>skainet-graph</artifactId>
        <version>0.1.0</version>
    </dependency>
    <!-- Add other modules as needed -->
</dependencies>
```

You'll need to configure your `~/.m2/settings.xml` with GitHub credentials:

```xml
<settings>
  <servers>
    <server>
      <id>github</id>
      <username>YOUR_GITHUB_USERNAME</username>
      <password>YOUR_GITHUB_TOKEN</password>
    </server>
  </servers>
</settings>
```

## Basic Concepts

SK-AI-Net is built around a few core concepts:

- **Tensors**: Multi-dimensional arrays that store data and support various operations
- **Compute Nodes**: Elements in a computation graph that perform operations on tensors
- **Modules**: Higher-level abstractions that represent neural network layers
- **Backends**: Implementations that handle the actual computations (CPU, GPU, etc.)

## Creating Your First Tensor

Here's how to create and manipulate tensors:

```kotlin
// Create a 2x3 tensor
val shape = Shape(2, 3)
val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
val tensor = DoublesTensor(shape, data)

// Access elements
val element = tensor[0, 1] // Returns 2.0

// Perform operations
val poweredTensor = tensor.pow(2.0) // Square each element
```

## Building a Simple Neural Network

SK-AI-Net makes it easy to build neural networks:

```kotlin
// Create a simple MLP with one hidden layer
val model = Sequential(
    Linear(inputSize = 784, outputSize = 128),
    ReLU(),
    Linear(inputSize = 128, outputSize = 10)
)

// Forward pass
fun predict(input: Tensor): Tensor {
    return model.forward(input)
}
```

## Loading Pre-trained Models

SK-AI-Net supports loading models from various formats:

```kotlin
// Load from SafeTensors format
val safeTensorsReader = SafeTensorsReader.fromFilePath("path/to/model.safetensors")
val tensorNames = safeTensorsReader.getTensorNames()
tensorNames.forEach { name ->
    val tensor = safeTensorsReader.readTensor(name)
    // Use the tensor in your model
}

// Load using a ParametersLoader
val loader = CsvParametersLoader { /* source provider */ }
loader.load { name, tensor ->
    // Map the tensor to your model's parameters
}
```

## Common Use Cases

### Image Classification

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
```

### Sequence Processing

```kotlin
// Example of processing a sequence of data
val sequence = listOf(tensor1, tensor2, tensor3)
val results = sequence.map { input ->
    model.forward(input)
}
```

## Next Steps

Now that you're familiar with the basics of SK-AI-Net, you can:

- Explore the [API documentation](https://sk-ai-net.github.io/skainet/dokka/) for detailed information on all classes and methods
- Check out the [examples directory](https://github.com/sk-ai-net/skainet/tree/main/samples) for more complex usage examples
- Join the [community forum](https://github.com/sk-ai-net/skainet/discussions) to ask questions and share your projects

For more advanced topics, refer to the following guides:
- [Custom Modules](./custom-modules.md)
- [Working with Different Backends](./backends.md)
- [Model Optimization](./optimization.md)
