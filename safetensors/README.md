# SafeTensor Module

This module provides functionality for reading [Hugging Face safetensor](https://huggingface.co/docs/safetensors/index) files in Kotlin Multiplatform projects.

## Overview

The safetensor format is a binary format for storing tensors, designed by Hugging Face for efficient and safe loading of machine learning models. It was created to address security and performance issues with existing formats like pickle and h5.

### Key Benefits of the Safetensor Format

- **Security**: Prevents deserialization vulnerabilities common in other formats
- **Performance**: Allows for memory mapping and fast random access to tensors
- **Simplicity**: Uses a straightforward binary format with a JSON header
- **Compatibility**: Works across different programming languages and platforms

### Format Structure

The safetensor format consists of three main parts:

1. An 8-byte header size (64-bit little-endian unsigned integer)
2. A JSON header containing metadata about the tensors (size specified by the header size)
3. The binary tensor data (with offsets specified in the JSON header)

![Safetensor Format Structure](../docs/images/safetensors-format.svg)

This module provides a `SafeTensorReader` interface and implementation for reading tensors from safetensor files.

## Usage

### Basic Usage

```kotlin
// Read a safetensor file
val fileBytes = File("model.safetensors").readBytes()
val reader = SafeTensorReader.fromByteArray(fileBytes)

// Get all tensor names
val tensorNames = reader.getTensorNames()
println("Available tensors: ${tensorNames.joinToString()}")

// Read a specific tensor
val tensor = reader.readTensor("model.weight")
if (tensor != null) {
    // Use the tensor
    println("Tensor shape: ${tensor.shape}")
    println("First value: ${tensor[0, 0]}")
}
```

### Advanced Usage

```kotlin
// Load a model from a safetensor file
val modelBytes = File("model.safetensors").readBytes()
val reader = SafeTensorReader.fromByteArray(modelBytes)

// Process all tensors in the file
reader.getTensorNames().forEach { name ->
    val tensor = reader.readTensor(name)
    if (tensor != null) {
        println("Processing tensor: $name with shape ${tensor.shape}")

        // Example: Calculate mean of tensor values
        var sum = 0.0
        var count = 0

        // For a 2D tensor
        if (tensor.shape.size == 2) {
            for (i in 0 until tensor.shape[0]) {
                for (j in 0 until tensor.shape[1]) {
                    sum += tensor[i, j]
                    count++
                }
            }
            val mean = if (count > 0) sum / count else 0.0
            println("Mean value of $name: $mean")
        }
    }
}
```

## Features

- Read tensors from safetensor files
- Support for different data types (F32, I8)
- Integration with the existing tensor system
- Memory-efficient loading of large safetensor files (on supported platforms)

## Memory Management

The safetensor format is designed to be memory-efficient, allowing for loading large models without excessive memory usage. This module provides several approaches for loading safetensor files, with different memory characteristics:

### Standard Loading (All Platforms)

The basic approach loads the entire safetensor file into memory:

```kotlin
// Load from a byte array (entire file in memory)
val reader = SafeTensorReader.fromByteArray(fileBytes)

// Load from a resource (entire file in memory)
val reader = SafeTensorReader.fromResource("model.safetensors")
```

This approach is simple but can be memory-intensive for large files, as it requires enough memory to hold the entire file.

### Memory-Mapped Loading (Desktop/JVM Only)

For large files, a memory-mapped approach is available on desktop/JVM platforms:

```kotlin
// Load from a file path using memory mapping
val reader = SafeTensorReader.fromFilePath("/path/to/model.safetensors")

// Load from a File object using memory mapping
val file = File("/path/to/model.safetensors")
val reader = SafeTensorReader.fromFile(file)
```

Memory-mapped loading offers several advantages for large files:

1. **Reduced Memory Usage**: Only the header and metadata are loaded into memory initially. Tensor data is mapped into memory when requested.

2. **Operating System Optimization**: The OS can efficiently manage memory, paging in only the parts of the file that are actually needed.

3. **Shared Memory**: Multiple processes can share the same memory-mapped file, reducing overall system memory usage.

4. **Faster Loading**: Initial loading is faster because only a small portion of the file needs to be read.

### Best Practices for Large Models

When working with large LLM models:

1. **Use Memory-Mapped Loading**: On desktop/JVM platforms, always use `fromFile` or `fromFilePath` for large models.

2. **Load Tensors Selectively**: Only load the tensors you need, when you need them.

3. **Release References**: Allow unused tensors to be garbage collected when no longer needed.

4. **Consider Quantization**: Use quantized models (e.g., with I8 data type) to reduce memory requirements.

## Implementation Details

### Format Parsing

The `SafeTensorReader` implementation follows these steps to parse a safetensor file:

1. **Read Header Size**: The first 8 bytes are read as a 64-bit little-endian unsigned integer to determine the size of the JSON header.

2. **Parse JSON Header**: The JSON header is extracted and parsed to obtain metadata about each tensor, including:
   - Data type (e.g., "F32", "I8")
   - Shape (dimensions of the tensor)
   - Data offsets (start and end positions of the tensor data)

3. **Read Tensor Data**: When a specific tensor is requested, its binary data is extracted from the file based on the offsets in the metadata, and converted to the appropriate tensor implementation.

### Memory-Mapped Implementation

The memory-mapped implementation (`MemoryMappedSafeTensorReader`) uses Java NIO's `MappedByteBuffer` to map the file into memory:

1. **Map File**: The file is memory-mapped using `FileChannel.map()`.

2. **Parse Header**: Only the header and metadata are fully loaded into memory.

3. **Access Tensor Data**: When a tensor is requested, a view of the mapped buffer is created for that tensor's data region, avoiding the need to load the entire tensor into memory at once.

### JSON Header Format

The JSON header has the following structure:

```json
{
  "tensor_name_1": {
    "dtype": "F32",
    "shape": [2, 3],
    "data_offsets": [0, 24]
  },
  "tensor_name_2": {
    "dtype": "I8",
    "shape": [10, 10],
    "data_offsets": [24, 124]
  }
}
```

### Supported Data Types

The current implementation supports the following data types:

- **F32**: 32-bit floating point values, converted to `SimpleTensor`
- **I8**: 8-bit integer values, converted to `Int8Tensor`

For unsupported data types, a `SimpleTensor` with zeros is returned.

### Integration with Tensor System

The `SafeTensorReader` provides access to the tensors through the `Tensor` interface, which allows seamless integration with the existing tensor system in the project. This enables operations like:

- Accessing tensor elements by indices
- Getting tensor shape information
- Using tensors in mathematical operations
