package sk.ai.net.io.gguf

import kotlinx.io.Source
import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.SimpleTensor
import sk.ai.net.graph.tensor.Int8Tensor
import sk.ai.net.graph.tensor.Int4Tensor
import sk.ai.net.graph.tensor.TernaryTensor
import sk.ai.net.io.ParametersLoader
import sk.ai.net.gguf.GGUFReader
import sk.ai.net.gguf.GGMLQuantizationType

/**
 * A parameters loader that loads tensors from a GGUF file.
 *
 * @param handleSource A function that returns a Source for the GGUF file.
 */
class GGUFParametersLoader(private val handleSource: () -> Source) : ParametersLoader {
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        handleSource().use { source ->
            val reader = GGUFReader(source)

            // Process each tensor in the GGUF file
            reader.tensors.forEach { tensor ->
                // Convert the tensor data to a DoublesTensor
                val shapeArray = tensor.shape.map { it.toInt() }.toIntArray()
                val shape = Shape(*shapeArray)

                // Convert the tensor data to a DoubleArray
                val doubleValues = when (tensor.tensorType) {
                    GGMLQuantizationType.F32 -> (tensor.data as List<*>).map { (it as Float).toDouble() }.toDoubleArray()
                    GGMLQuantizationType.F64 -> (tensor.data as List<*>).map { it as Double }.toDoubleArray()
                    GGMLQuantizationType.I8 -> (tensor.data as List<*>).map { (it as Byte).toDouble() }.toDoubleArray()
                    GGMLQuantizationType.I16 -> (tensor.data as List<*>).map { (it as Short).toDouble() }.toDoubleArray()
                    GGMLQuantizationType.I32 -> (tensor.data as List<*>).map { (it as Int).toDouble() }.toDoubleArray()
                    GGMLQuantizationType.I64 -> (tensor.data as List<*>).map { (it as Long).toDouble() }.toDoubleArray()
                    else -> throw IllegalArgumentException("Unsupported tensor type: ${tensor.tensorType}")
                }

                // Create the appropriate tensor type based on the quantization type
                val resultTensor: Tensor = when (tensor.tensorType) {
                    GGMLQuantizationType.I8 -> {
                        // For I8, use Int8Tensor with the original byte data if possible
                        if (tensor.data is List<*> && tensor.data.isNotEmpty() && tensor.data.first() is Byte) {
                            val byteData = (tensor.data as List<*>).map { it as Byte }.toByteArray()
                            // Use a default scale and zero point for now
                            Int8Tensor(shape, byteData, 1.0, 0)
                        } else {
                            // Fallback to SimpleTensor if we can't get the raw byte data
                            SimpleTensor(shape, doubleValues)
                        }
                    }
                    GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q4_1, GGMLQuantizationType.Q4_K, GGMLQuantizationType.IQ4_NL, GGMLQuantizationType.IQ4_XS -> {
                        // For 4-bit quantization types, use Int4Tensor.fromDoubles to properly quantize the values
                        Int4Tensor.fromDoubles(shape, doubleValues)
                    }
                    GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0 -> {
                        // For ternary quantization types, use TernaryTensor
                        TernaryTensor.fromDoubles(shape, doubleValues)
                    }
                    else -> {
                        // For other types, use SimpleTensor
                        SimpleTensor(shape, doubleValues)
                    }
                }

                // Call the callback with the tensor name and the tensor
                onTensorLoaded(tensor.name, resultTensor)
            }
        }
    }
}
