package sk.ai.net.core.tensor

import sk.ai.net.core.backend.BackendInitializer

/**
 * Initializes the backend for the desktop platform.
 */
fun initializeBackend() {
    BackendInitializer.initialize()
}

// Initialize the backend when this file is loaded
private val initialized = run {
    initializeBackend()
    true
}

/**
 * Desktop implementation of tensor multiplication.
 */
actual operator fun Tensor.times(other: Tensor): Tensor = defaultTimes(this, other)

/**
 * Desktop implementation of tensor addition.
 */
actual operator fun Tensor.plus(other: Tensor): Tensor = defaultPlus(this, other)

/**
 * Desktop implementation of matrix multiplication.
 */
actual infix fun Tensor.matmul(other: Tensor): Tensor = defaultMatmul(this, other)

/**
 * Desktop implementation of ReLU activation.
 */
actual fun relu(tensor: Tensor): Tensor = defaultRelu(tensor)
