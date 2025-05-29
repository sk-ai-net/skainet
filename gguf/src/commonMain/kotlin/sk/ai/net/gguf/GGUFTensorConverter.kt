package sk.ai.net.gguf

import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.shape.Shape

/**
 * Utility class for converting GGUF tensors to graph tensors.
 */
object GGUFTensorConverter {

    /**
     * Converts a GGUF ReaderTensor to a graph Tensor.
     *
     * @param readerTensor The GGUF tensor to convert.
     * @return The converted graph tensor.
     * @throws IllegalArgumentException If the tensor type is not supported.
     */
    fun convert(readerTensor: ReaderTensor): Tensor {
        // Convert the shape
        val dimensions = readerTensor.shape.map { it.toInt() }
        val shape = Shape(dimensions.toIntArray())

        // Create a double array to hold the tensor data
        val size = readerTensor.nElements
        val doubleData = DoubleArray(size)

        // Convert the data based on the tensor type and structure
        when (readerTensor.tensorType) {
            GGMLQuantizationType.F32 -> {
                // Handle different possible structures of float data
                when (val data = readerTensor.data) {
                    is List<*> -> {
                        // Handle flat list of floats
                        if (data.isNotEmpty() && data[0] is Float) {
                            for (i in data.indices) {
                                doubleData[i] = (data[i] as Float).toDouble()
                            }
                        } 
                        // Handle nested lists (e.g., for matrices)
                        else if (data.isNotEmpty() && data[0] is List<*>) {
                            var index = 0
                            for (row in data) {
                                if (row is List<*>) {
                                    for (element in row) {
                                        if (element is Float) {
                                            doubleData[index++] = element.toDouble()
                                        } else if (element is Number) {
                                            doubleData[index++] = element.toDouble()
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Handle other possible data structures
                    else -> {
                        throw IllegalArgumentException("Unsupported data structure for F32 tensor: ${data::class.simpleName}")
                    }
                }
            }
            GGMLQuantizationType.F64 -> {
                // Handle different possible structures of double data
                when (val data = readerTensor.data) {
                    is List<*> -> {
                        // Handle flat list of doubles
                        if (data.isNotEmpty() && data[0] is Double) {
                            for (i in data.indices) {
                                doubleData[i] = data[i] as Double
                            }
                        } 
                        // Handle nested lists (e.g., for matrices)
                        else if (data.isNotEmpty() && data[0] is List<*>) {
                            var index = 0
                            for (row in data) {
                                if (row is List<*>) {
                                    for (element in row) {
                                        if (element is Double) {
                                            doubleData[index++] = element
                                        } else if (element is Number) {
                                            doubleData[index++] = element.toDouble()
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Handle other possible data structures
                    else -> {
                        throw IllegalArgumentException("Unsupported data structure for F64 tensor: ${data::class.simpleName}")
                    }
                }
            }
            GGMLQuantizationType.I8, GGMLQuantizationType.I16, GGMLQuantizationType.I32, GGMLQuantizationType.I64 -> {
                // Handle different possible structures of integer data
                when (val data = readerTensor.data) {
                    is List<*> -> {
                        // Handle flat list of integers
                        if (data.isNotEmpty() && data[0] is Number) {
                            for (i in data.indices) {
                                doubleData[i] = (data[i] as Number).toDouble()
                            }
                        } 
                        // Handle nested lists (e.g., for matrices)
                        else if (data.isNotEmpty() && data[0] is List<*>) {
                            var index = 0
                            for (row in data) {
                                if (row is List<*>) {
                                    for (element in row) {
                                        if (element is Number) {
                                            doubleData[index++] = element.toDouble()
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Handle other possible data structures
                    else -> {
                        throw IllegalArgumentException("Unsupported data structure for integer tensor: ${data::class.simpleName}")
                    }
                }
            }
            else -> throw IllegalArgumentException("Unsupported tensor type: ${readerTensor.tensorType}")
        }

        return SimpleTensor(shape, doubleData)
    }
}
