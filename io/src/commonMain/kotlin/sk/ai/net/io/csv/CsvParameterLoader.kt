package sk.ai.net.io.csv

import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.ParametersLoader

/**
 * A parameters loader that loads tensors from a CSV file.
 *
 * This implementation reads a JSON-formatted CSV file containing tensor parameters
 * and converts them to tensor objects that can be used by the model.
 *
 * @param handleSource A function that returns a Source for the CSV file.
 */
class CsvParametersLoader(private val handleSource: () -> Source) :
    ParametersLoader {
    /**
     * Loads parameters from the CSV file and invokes the provided callback for each tensor loaded.
     *
     * This method reads the CSV file, parses the JSON content, and converts each parameter
     * to a tensor. For each tensor, it calls the provided callback function with the
     * parameter name and the tensor.
     *
     * @param onTensorLoaded A callback function that is invoked for each tensor loaded.
     *                       The callback receives the tensor name and the tensor itself.
     */
    override suspend fun load(onTensorLoaded: (String, Tensor) -> Unit) {
        handleSource().use { source: Source ->
            // Initialize Json object
            val json = Json { ignoreUnknownKeys = true }
            // Deserialize JSON to Kotlin objects

            json.decodeFromString<List<Parameter>>(source.readString()).also { values ->
                values.forEach { parameter ->
                    val tensor = DoublesTensor(Shape(*parameter.tensor.shape.toIntArray()), parameter.tensor.values.toDoubleArray())
                    onTensorLoaded(parameter.unique_parameter_name, tensor)
                }
            }
        }
    }
}
