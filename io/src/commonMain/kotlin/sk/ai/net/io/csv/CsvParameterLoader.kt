package sk.ai.net.io.csv

import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlinx.io.readString
import kotlinx.serialization.json.Json
import sk.ai.net.Shape

import sk.ai.net.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.io.ParametersLoader

class CsvParametersLoader(private val handleSource: () -> Source) :
    ParametersLoader {
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
