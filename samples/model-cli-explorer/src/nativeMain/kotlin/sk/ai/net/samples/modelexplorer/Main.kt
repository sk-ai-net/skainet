package sk.ai.net.samples.modelexplorer

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.required
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray

/**
 * Native-specific main entry point for the model explorer CLI application.
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
        val fileSystem = SystemFileSystem
        val file = fileSystem.openReadOnly(Path(filePath))

        val fileSize = file.size()
        println("File Size: ${formatFileSize(fileSize)}")
        println()

        // Read the first few bytes to detect the file format
        val bytes = ByteArray(4)
        file.read(bytes, 0, 4)

        // Detect file format
        val format = when {
            // Check for GGUF format (magic number: 0x46554747 or "GGUF" in ASCII)
            bytes[0].toInt() == 0x47 && // 'G'
            bytes[1].toInt() == 0x47 && // 'G'
            bytes[2].toInt() == 0x55 && // 'U'
            bytes[3].toInt() == 0x46 -> "GGUF" // 'F'

            // Check for SafeTensors format (JSON header at the beginning)
            bytes[0].toInt() == 0x7B && bytes[1].toInt() == 0x22 -> "SafeTensors" // '{' and '"'

            // Unknown format
            else -> "Unknown"
        }

        println("File Format: $format")
        println()

        // Print metadata and statistics
        println("Metadata:")
        println("  Format: $format")
        println("  Size: ${formatFileSize(fileSize)}")
        println()

        println("Statistics:")
        println("  File Size: ${formatFileSize(fileSize)}")
        println("  Format: $format")
        println("  Number of Tensors: N/A (simplified implementation)")
        println("  Memory Consumption: N/A (simplified implementation)")
        println()

        println("Tensors:")
        println("  This is a simplified implementation. For full tensor information,")
        println("  please use the complete implementation with the ModelFormat API.")

        // Close the file
        file.close()
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

/**
 * Formats a number with thousands separators.
 *
 * @param number The number to format.
 * @return A string representation of the number with thousands separators.
 */
fun formatNumber(number: Int): String {
    return number.toString().reversed().chunked(3).joinToString(",").reversed()
}
