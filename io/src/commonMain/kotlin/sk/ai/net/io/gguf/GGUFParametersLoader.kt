package sk.ai.net.io.gguf

import kotlinx.io.Source
import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
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

                // Create a DoublesTensor with the shape and values
                val doublesTensor = DoublesTensor(shape, doubleValues)

                // Call the callback with the tensor name and the DoublesTensor
                onTensorLoaded(tensor.name, doublesTensor)
            }
        }
    }
}
