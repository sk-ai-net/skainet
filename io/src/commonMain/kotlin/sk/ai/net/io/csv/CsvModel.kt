package sk.ai.net.io.csv

import kotlinx.serialization.Serializable

/**
 * Represents a tensor in the CSV format.
 *
 * This data class is used for serialization and deserialization of tensor data
 * when loading parameters from CSV files.
 *
 * @property shape The shape of the tensor as a list of dimensions.
 * @property values The values of the tensor as a list of doubles.
 */
@Serializable
data class Tensor(
    val shape: List<Int>,
    val values: List<Double>
)

/**
 * Represents a named parameter in the CSV format.
 *
 * This data class associates a unique name with a tensor, allowing for
 * identification and retrieval of specific parameters in a model.
 *
 * @property unique_parameter_name The unique identifier for this parameter.
 * @property tensor The tensor data associated with this parameter.
 */
@Serializable
data class Parameter(
    val unique_parameter_name: String,
    val tensor: Tensor
)
