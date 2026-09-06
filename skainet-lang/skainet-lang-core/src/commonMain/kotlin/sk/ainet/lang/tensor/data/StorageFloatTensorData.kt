package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * Dense FP32 tensor data over a [Storage.Heap] region — the creation-path end of the Scope split
 * (#1145): what `ExecutionContext.zeros/full/fromFloatArray` hand out when a [sk.ainet.lang.memory.Scope]
 * other than `Ambient` is active, so an activation's bytes come from the forward slab and die at
 * `reset()` instead of waiting for the GC.
 *
 * Every access goes through [Storage.checkAlive], so a use-after-reset is a
 * [sk.ainet.lang.memory.StorageClosedException] naming the storage — not silent corruption.
 *
 * **Deliberately not a [FloatArrayTensorData].** A slab slice has a nonzero
 * [Storage.Heap.arrayOffset], and the ops fast paths that unwrap `buffer` assume offset 0; exposing
 * this data through that interface would hand them the whole slab. Element access and
 * [copyToFloatArray] are offset-correct; kernels that want zero-copy take [view], which carries the
 * offset properly.
 */
public class StorageFloatTensorData<T : DType>(
    initialShape: Shape,
    public val storage: Storage.Heap,
) : TensorData<T, Float> {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()

    init {
        requireNotNull(storage.floats) { "StorageFloatTensorData needs float-backed storage" }
        require(storage.elementCount >= shape.volume) {
            "storage holds ${storage.elementCount} floats, shape $shape needs ${shape.volume}"
        }
    }

    private fun flatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flat = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flat += idx * strides[i]
        }
        return flat
    }

    override fun get(vararg indices: Int): Float {
        storage.checkAlive()
        return storage.floats!![storage.arrayOffset + flatIndex(indices)]
    }

    override fun set(vararg indices: Int, value: Float) {
        storage.checkAlive()
        storage.floats!![storage.arrayOffset + flatIndex(indices)] = value
    }

    override fun copyToFloatArray(): FloatArray {
        storage.checkAlive()
        val off = storage.arrayOffset
        return storage.floats!!.copyOfRange(off, off + shape.volume)
    }

    /** A dense view over the same region — the storage carries the offset, nothing is copied. */
    override val view: TensorView
        get() {
            storage.checkAlive()
            return TensorView.dense(storage, shape, FP32)
        }
}
