package sk.ai.net.samples.mnist.mlp

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSource
import kotlinx.io.buffered
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.dsl.network
import sk.ai.net.gguf.GGMLQuantizationType
import sk.ai.net.gguf.GGUFReader
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.data.mnist.MNISTConstants
import sk.ai.net.io.data.mnist.MNISTImage
import sk.ai.net.io.data.mnist.MNISTLoaderFactory
import sk.ai.net.nn.Linear
import sk.ai.net.nn.Module
import sk.ai.net.nn.activations.ReLU
import java.io.File
import java.io.FileInputStream


/**
 * Main entry point for the mnist-mlp-cli application.
 *
 * This application creates an MLP network, loads model weights from a GGUF file,
 * and uses the model to classify MNIST handwritten digits.
 * It loads the MNIST test data, classifies a subset of the images, and reports the accuracy.
 *
 * Usage:
 * ```
 * ./gradlew :samples:mnist-mlp-cli:run --args="-m model.gguf -n 100"
 * ```
 *
 * Options:
 * - `-m, --model`: Path to the model file in GGUF format (required)
 * - `-n, --num-images`: Number of images to classify (default: 10)
 */
actual fun main(args: Array<String>) {
    val parser = ArgParser("mnist-mlp-cli")
    val modelPath by parser.option(
        ArgType.String,
        shortName = "m",
        fullName = "model",
        description = "Path to the model file (GGUF format)"
    ).required()
    val numImagesArg by parser.option(
        ArgType.Int,
        shortName = "n",
        fullName = "num-images",
        description = "Number of images to classify (default: 10)"
    )
    val numImages = numImagesArg ?: 10

    parser.parse(args)

    println("MNIST MLP CLI")
    println("=============")
    println("Model file: $modelPath")
    println("Number of images to classify: $numImages")
    println()

    try {
        // Create the MLP network
        println("Creating MLP network...")
        val mlp = createMNISTMLP()

        // Load model weights from GGUF file
        println("Loading model weights from $modelPath...")
        loadModelWeights(mlp, modelPath)

        // Load MNIST test data
        println("Loading MNIST test data...")
        val testData = runBlocking {
            val loader = MNISTLoaderFactory.create()
            loader.loadTestData()
        }
        println("Loaded ${testData.size} test images")

        // Classify a subset of the test images
        val numImagesToClassify = minOf(numImages ?: 10, testData.size)
        println("\nClassifying $numImagesToClassify images...")

        var correctPredictions = 0
        for (i in 0 until numImagesToClassify) {
            val image = testData.images[i]
            val prediction = classifyImage(mlp, image)
            val actualLabel = image.label.toInt()
            val isCorrect = prediction == actualLabel

            if (isCorrect) {
                correctPredictions++
            }

            println("Image ${i + 1}: Predicted $prediction, Actual $actualLabel ${if (isCorrect) "✓" else "✗"}")
        }

        // Print accuracy
        val accuracy = correctPredictions.toDouble() / numImagesToClassify * 100
        println("\nAccuracy: $correctPredictions/$numImagesToClassify (${accuracy.toInt()}%)")

    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

/**
 * Creates an MLP network for MNIST digit classification.
 *
 * The network architecture is as follows:
 * - Input layer: 784 neurons (28x28 pixels flattened)
 * - First hidden layer: 128 neurons with ReLU activation
 * - Second hidden layer: 128 neurons with ReLU activation
 * - Output layer: 10 neurons (one for each digit 0-9)
 *
 * This is a simple feedforward neural network that takes a flattened MNIST image as input
 * and outputs a probability distribution over the 10 possible digits. The digit with the
 * highest probability is the predicted digit.
 *
 * @return The MLP network.
 */
private fun createMNISTMLP(): Module {
    // Create the MLP network using the DSL
    return network {
        input(MNISTConstants.IMAGE_PIXELS) // 784 input neurons (28x28 pixels)
        dense(128, "hidden1") { // First hidden layer with 128 neurons
            activation = ReLU()::invoke
        }
        dense(10, "output") { // Output layer with 10 neurons (one for each digit 0-9)
            // No activation for the output layer (linear)
        }
    }
}

/**
 * Converts a MNIST image to a tensor suitable for input to the MLP network.
 *
 * The MNIST image is a 28x28 grayscale image stored as a ByteArray.
 * This function:
 * 1. Normalizes the pixel values from [0, 255] to [0, 1]
 * 2. Flattens the 28x28 image into a 784-element vector
 * 3. Creates a tensor with shape [784]
 *
 * @param image The MNIST image to convert.
 * @return A tensor representing the image.
 */
private fun convertImageToTensor(image: MNISTImage): Tensor {
    // Normalize pixel values to range [0, 1]
    val normalizedPixels = DoubleArray(MNISTConstants.IMAGE_PIXELS) { i ->
        (image.image[i].toInt() and 0xFF) / 255.0
    }

    // Create a tensor with shape [784] (flattened 28x28 image)
    return DoublesTensor(Shape(MNISTConstants.IMAGE_PIXELS), normalizedPixels)
}

/**
 * Loads model weights from a GGUF file and applies them to the MLP network.
 *
 * This function:
 * 1. Opens the GGUF file
 * 2. Extracts the tensors
 * 3. Applies them to the MLP network
 *
 * @param mlp The MLP network to apply the weights to.
 * @param modelPath The path to the GGUF file.
 */
private fun loadModelWeights(mlp: Module, modelPath: String) {
    // Open the GGUF file
    val inputStream = FileInputStream(File(modelPath))

    // Create a GGUFReader to parse the file
    val reader = GGUFReader(inputStream.asSource().buffered())

    // Get the tensors from the GGUF file
    val tensors = reader.tensors

    println("Found ${tensors.size} tensors in the GGUF file")

    // Apply the tensors to the MLP network
    // The MLP network has the following structure:
    // - Input layer
    // - Linear layer 1 (784 -> 128) + ReLU
    // - Linear layer 2 (128 -> 128) + ReLU
    // - Linear layer 3 (128 -> 10)

    // Get the linear layers from the MLP
    val linearLayers = mutableListOf<Linear>()
    fun findLinearLayers(module: Module) {
        if (module is Linear) {
            linearLayers.add(module)
        }
        for (child in module.modules) {
            findLinearLayers(child)
        }
    }
    findLinearLayers(mlp)

    for (layer in linearLayers) {
        val name = layer.name
        println("Processing tensor: $name")
    }


    println("Found ${linearLayers.size} linear layers in the MLP network")

    // Map the tensors to the linear layers
    for (tensor in tensors) {
        val name = tensor.name.substring("model.".length)
        println("Processing tensor: $name")

        // Find the corresponding linear layer and parameter
        for (layer in linearLayers) {
            for (param in layer.params) {
                if (param.name.contains(name, ignoreCase = true)) {
                    // Convert the tensor data to a DoublesTensor
                    val tensor = when (tensor.tensorType) {
                        GGMLQuantizationType.F32,
                        GGMLQuantizationType.F16 -> {
                            val floatsList = tensor.data as List<Float>
                            DoublesTensor(
                                param.value.shape,
                                floatsList.map { it.toDouble() }.toDoubleArray()
                            )
                        }
                        else -> {
                            println("Unsupported tensor data type: ${tensor.data.firstOrNull()?.javaClass}")
                            continue
                        }
                    }

                    // Apply the tensor to the parameter
                    param.value = tensor
                    println("Applied tensor $name to parameter ${param.name}")
                }
            }
        }
    }
}

/**
 * Classifies a MNIST image using the given MLP network.
 *
 * This function:
 * 1. Converts the MNIST image to a tensor
 * 2. Passes the tensor through the MLP network
 * 3. Finds the index of the maximum value in the output tensor
 * 4. Returns the index as the predicted digit (0-9)
 *
 * The output tensor has 10 elements, one for each possible digit.
 * The element with the highest value corresponds to the most likely digit.
 *
 * @param mlp The MLP network to use for classification.
 * @param image The MNIST image to classify.
 * @return The predicted digit (0-9).
 */
private fun classifyImage(mlp: Module, image: MNISTImage): Int {
    // Convert the image to a tensor
    val inputTensor = convertImageToTensor(image)

    // Forward pass through the network
    val outputTensor = mlp.forward(inputTensor) as DoublesTensor

    // Find the index of the maximum value in the output tensor
    // This is the predicted digit
    val outputElements = outputTensor.elements
    var maxIndex = 0
    var maxValue: Double = outputElements[0]

    for (i in 1 until outputElements.size) {
        if (outputElements[i] > maxValue) {
            maxValue = outputElements[i]
            maxIndex = i
        }
    }

    return maxIndex
}
