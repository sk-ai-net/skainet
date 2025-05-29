package sk.ai.net.safetensors

import java.io.File

/**
 * Desktop-specific implementations for SafeTensorReader.
 */

// Override the companion object methods with desktop-specific implementations
// that use memory-mapped files for efficient loading of large safetensor files

/**
 * Desktop-specific implementation of SafeTensorReader.fromFilePath.
 *
 * This implementation uses memory-mapped files for efficient loading of large safetensor files.
 * It only loads the header and metadata into memory, and maps the tensor data into memory when requested.
 *
 * @param filePath The path to the file containing the safetensor data.
 * @return A SafeTensorReader instance.
 */
@Suppress("EXTENSION_SHADOWED_BY_MEMBER")
fun SafeTensorsReader.Companion.fromFilePath(filePath: String): SafeTensorsReader {
    return MemoryMappedSafeTensorsReader.fromFilePath(filePath)
}

/**
 * Desktop-specific implementation of SafeTensorReader.fromFile.
 *
 * This implementation uses memory-mapped files for efficient loading of large safetensor files.
 * It only loads the header and metadata into memory, and maps the tensor data into memory when requested.
 *
 * @param file The file containing the safetensor data.
 * @return A SafeTensorReader instance.
 */
@Suppress("EXTENSION_SHADOWED_BY_MEMBER", "UNCHECKED_CAST")
fun SafeTensorsReader.Companion.fromFile(file: Any): SafeTensorsReader {
    if (file !is File) {
        throw IllegalArgumentException("Expected java.io.File, got ${file::class.java.name}")
    }
    return MemoryMappedSafeTensorsReader.fromFile(file)
}

// Keep the top-level functions for backward compatibility with existing tests
/**
 * Creates a SafeTensorReader from the given file path.
 *
 * This implementation uses memory-mapped files for efficient loading of large safetensor files.
 * It only loads the header and metadata into memory, and maps the tensor data into memory when requested.
 *
 * @param filePath The path to the file containing the safetensor data.
 * @return A SafeTensorReader instance.
 */
fun fromFilePath(filePath: String): SafeTensorsReader {
    return MemoryMappedSafeTensorsReader.fromFilePath(filePath)
}

/**
 * Creates a SafeTensorReader from the given file.
 *
 * This implementation uses memory-mapped files for efficient loading of large safetensor files.
 * It only loads the header and metadata into memory, and maps the tensor data into memory when requested.
 *
 * @param file The file containing the safetensor data.
 * @return A SafeTensorReader instance.
 */
fun fromFile(file: File): SafeTensorsReader {
    return MemoryMappedSafeTensorsReader.fromFile(file)
}
