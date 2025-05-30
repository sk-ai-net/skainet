package sk.ai.net.samples.modelexplorer

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import kotlinx.coroutines.runBlocking
import kotlinx.io.asSource
import kotlinx.io.buffered
import java.io.File
import java.io.FileInputStream
import sk.ai.net.io.model.ModelFormat
import sk.ai.net.io.model.ModelFormatLoader
import sk.ai.net.io.model.ModelFormatType

/**
 * JVM-specific main entry point for the model explorer CLI application.
 */
actual fun main(args: Array<String>) {
    val parser = ArgParser("model-explorer")
    val filePath by parser.option(
        ArgType.String,
        shortName = "f",
        fullName = "file",
        description = "Path to the model file"
    ).required()

    parser.parse(args)

    println("Model Explorer")
    println("==============")
    println("File: $filePath")
    println()

    try {
        // Open the file
        val file = File(filePath)
        if (!file.exists()) {
            println("Error: File not found: $filePath")
            return
        }

        val fileSize = file.length()
        println("File Size: ${formatFileSize(fileSize)}")
        println()

        // Create a function that returns a fresh source each time it's called
        val sourceProvider = { FileInputStream(file).asSource().buffered() }

        val loader = ModelFormatLoader(sourceProvider)
        runBlocking {
            println("Detecting model format...")
            val formatType = loader.modelFormatType
            println("Model format detected: $formatType")

            println("Getting metadata...")
            val metadata = loader.getMetadata()

            // Print metadata and statistics
            println("Metadata:")
            println("  Format: $formatType")
            println("  Size: ${formatFileSize(fileSize)}")
            println("  Metadata: $metadata")

            // Print tensor names to help diagnose the issue
            println("Getting tensor names...")
            val tensorNames = loader.getTensorNames()
            println("Tensor names: $tensorNames")
            println()

            println("Statistics:")
            println("  Number of Tensors: ${tensorNames.size}")
            println()

            // Get and display detailed tensor metadata
            println("Tensors:")
            if (tensorNames.isNotEmpty()) {
                // For each tensor, get its metadata and display it
                tensorNames.forEachIndexed { index, name ->
                    val tensor = loader.getTensor(name)
                    if (tensor != null) {
                        println("  $index. $name")
                        println("    Shape: ${tensor.shape.dimensions.joinToString(", ")}")
                        println("    Data Type: ${when (formatType) {
                            ModelFormatType.GGUF -> "GGUF tensor type"
                            ModelFormatType.SAFETENSORS -> "SafeTensors tensor type"
                            else -> "Unknown"
                        }}")
                        println("    Size: ${formatFileSize((tensor.shape.volume * when (formatType) {
                            ModelFormatType.GGUF -> 4 // Assuming 4 bytes per element for GGUF
                            ModelFormatType.SAFETENSORS -> 4 // Assuming 4 bytes per element for SafeTensors
                            else -> 4 // Default to 4 bytes per element
                        }).toLong())}")
                    }
                }
            } else {
                println("  No tensors found in the model file.")
            }
        }

        // This is now commented out as we've implemented the tensor metadata display above
        // println("Tensors:")
        // println("  This is a simplified implementation. For full tensor information,")
        // println("  please use the complete implementation with the ModelFormat API.")

    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}

/**
 * Formats a size in bytes to a human-readable string.
 *
 * @param bytes The size in bytes.
 * @return A human-readable string representation of the size.
 */
fun formatFileSize(bytes: Long): String {
    val units = arrayOf("B", "KB", "MB", "GB", "TB")
    var size = bytes.toDouble()
    var unitIndex = 0

    while (size >= 1024 && unitIndex < units.size - 1) {
        size /= 1024
        unitIndex++
    }

    return "%.2f %s".format(size, units[unitIndex])
}
