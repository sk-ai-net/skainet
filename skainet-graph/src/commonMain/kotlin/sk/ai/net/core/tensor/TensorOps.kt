package sk.ai.net.core.tensor

import sk.ai.net.core.backend.BackendFactory

/**
 * Multiplies two tensors element-wise.
 *
 * @param other The tensor to multiply with.
 * @return The result of multiplying the two tensors.
 */
expect operator fun Tensor.times(other: Tensor): Tensor

/**
 * Adds two tensors element-wise.
 *
 * @param other The tensor to add.
 * @return The result of adding the two tensors.
 */
expect operator fun Tensor.plus(other: Tensor): Tensor

/**
 * Performs matrix multiplication of two tensors.
 *
 * @param other The tensor to multiply with.
 * @return The result of matrix multiplication.
 */
expect infix fun Tensor.matmul(other: Tensor): Tensor

/**
 * Applies the ReLU activation function to a tensor.
 *
 * @param tensor The input tensor.
 * @return The result of applying ReLU.
 */
expect fun relu(tensor: Tensor): Tensor

// Default implementations that can be used by platform-specific code
internal fun defaultTimes(tensor: Tensor, other: Tensor): Tensor =
    BackendFactory.getDefaultBackend().multiply(tensor, other)

internal fun defaultPlus(tensor: Tensor, other: Tensor): Tensor =
    BackendFactory.getDefaultBackend().add(tensor, other)

internal fun defaultMatmul(tensor: Tensor, other: Tensor): Tensor =
    BackendFactory.getDefaultBackend().matmul(tensor, other)

internal fun defaultRelu(tensor: Tensor): Tensor =
    BackendFactory.getDefaultBackend().relu(tensor)
