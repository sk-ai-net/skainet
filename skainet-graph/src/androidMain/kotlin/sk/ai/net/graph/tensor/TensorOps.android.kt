package sk.ai.net.graph.tensor

import sk.ai.net.graph.backend.AndroidBackendInitializer
import sk.ai.net.graph.backend.BackendFactory
import sk.ai.net.graph.backend.AndroidCpuBackend

/**
 * Initializes the backend for the Android platform.
 */
fun initializeBackend() {
    AndroidBackendInitializer.initialize()
}

// Initialize the backend when this file is loaded
private val initialized = run {
    initializeBackend()
    true
}

/**
 * Android implementation of tensor multiplication.
 */
actual operator fun Tensor.times(other: Tensor): Tensor = defaultTimes(this, other)

/**
 * Android implementation of tensor addition.
 */
actual operator fun Tensor.plus(other: Tensor): Tensor = defaultPlus(this, other)

/**
 * Android implementation of matrix multiplication.
 */
actual infix fun Tensor.matmul(other: Tensor): Tensor = defaultMatmul(this, other)

/**
 * Android implementation of ReLU activation.
 */
actual fun relu(tensor: Tensor): Tensor = defaultRelu(tensor)
