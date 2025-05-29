package sk.ai.net.core

import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.core.Slice

/**
 * Interface for tensors that store a specific type of data.
 *
 * This interface extends the basic Tensor interface and adds type information.
 * It's used for tensors that store a specific type of data, such as Double, Int, etc.
 *
 * @param T The type of data stored in the tensor.
 */
interface TypedTensor<T> : Tensor {
    /**
     * Retrieves a slice of the tensor.
     *
     * @param slices The slices to apply to the tensor.
     * @return A new tensor containing the sliced data.
     */
    operator fun get(vararg slices: Slice): TypedTensor<T>
}
