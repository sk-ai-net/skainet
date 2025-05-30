package sk.ai.net.core.tensor

import sk.ai.net.core.backend.WasmBackendInitializer

/**
 * Initializes the backend for the WebAssembly platform.
 */
fun initializeBackend() {
    WasmBackendInitializer.initialize()
}

// Initialize the backend when this file is loaded
private val initialized = run {
    initializeBackend()
    true
}

/**
 * WebAssembly implementation of tensor multiplication.
 */
actual operator fun Tensor.times(other: Tensor): Tensor = defaultTimes(this, other)

/**
 * WebAssembly implementation of tensor addition.
 */
actual operator fun Tensor.plus(other: Tensor): Tensor = defaultPlus(this, other)

/**
 * WebAssembly implementation of matrix multiplication.
 */
actual infix fun Tensor.matmul(other: Tensor): Tensor = defaultMatmul(this, other)

/**
 * WebAssembly implementation of ReLU activation.
 */
actual fun relu(tensor: Tensor): Tensor = defaultRelu(tensor)
