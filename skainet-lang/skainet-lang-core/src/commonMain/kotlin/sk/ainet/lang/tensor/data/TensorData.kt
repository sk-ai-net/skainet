package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * A fundamental data structure interface that provides indexed access to elements.
 * 
 * This interface serves as the base for all data structures that need to provide
 * element access through multidimensional indexing. It is designed to support
 * efficient access patterns commonly used in neural network computations.
 * 
 * @param T the type of elements that can be accessed
 */
public interface ItemsAccessor<T> {
    /**
     * Retrieves an element at the specified multidimensional indices.
     * 
     * This operator function allows accessing elements using bracket notation
     * with variable number of indices, supporting tensors of any dimensionality.
     * 
     * @param indices the coordinates to access the element at each dimension
     * @return the element of type T at the specified indices
     * @throws IndexOutOfBoundsException if any index is out of bounds
     */
    public operator fun get(vararg indices: Int): T


    /**
     * Setter
     */
    public operator fun set(vararg indices: Int, value: T)
}

/**
 * The fundamental data structure for tensor operations in the SKaiNET framework.
 * 
 * TensorData represents the core abstraction for all tensor-like data structures
 * used throughout the neural network computation system. It combines element access
 * capabilities with shape information, providing a unified interface for working
 * with multi-dimensional data arrays.
 * 
 * This interface serves as the foundation for:
 * - Neural network weight storage
 * - Activation value containers  
 * - Gradient computation data structures
 * - Input/output tensor representations
 * 
 * The generic type parameters allow for flexible data type support while maintaining
 * type safety across different numerical precisions and value representations.
 * 
 * @param T the data type constraint extending DType, defining the numerical precision
 * @param V the actual value type that will be stored and accessed
 */
public interface TensorData<T : DType, V> : ItemsAccessor<V> {
    /**
     * The shape descriptor that defines the dimensionality and size of this tensor data.
     *
     * The shape property provides essential metadata about the tensor's structure,
     * including the number of dimensions and the size along each dimension. This
     * information is crucial for:
     * - Bounds checking during element access
     * - Memory layout calculations
     * - Broadcasting operations
     * - Tensor operation compatibility verification
     *
     * @return the Shape object describing this tensor's dimensional structure
     */
    public val shape: Shape

    /**
     * How this data's bytes are laid out, or `null` when they are the plain dense representation
     * of the tensor's logical dtype (the default for array-backed data).
     *
     * Packed implementations (Q4_0 … Q8_0, Q4_K/Q5_K/Q6_K, ternary) report their block encoding;
     * narrow-float data reports `Dense(2)`. Together with the tensor's dtype witness this yields
     * the tensor's `Format` (SKEEP-003 rule 3: a packed weight is *logically* FP32 — the encoding
     * says how it is stored, it never replaces the dtype). A default member, not an abstract one,
     * because `TensorData` is implemented outside this module.
     */
    public val encoding: sk.ainet.lang.tensor.storage.TensorEncoding? get() = null

    /**
     * This data as a [sk.ainet.lang.memory.TensorView] — `Shape + Format + Layout + Storage` — or
     * `null` when the implementation cannot expose one (SKEEP-003 §4.1: `TensorData` becomes a
     * façade over the view; migrated kernels take the view, everything else keeps using this
     * interface unchanged).
     *
     * The view is over the *same* bytes: for array-backed data the storage borrows the array
     * (`Storage.Heap.wrap`), so writes through either side are visible on both and nothing is
     * copied. Per-element access stays on this interface's own fast path — the Phase-2 spike
     * (#1016) showed a view is for unwrapping once per call, not for per-element reads.
     */
    public val view: sk.ainet.lang.memory.TensorView? get() = null

    /**
     * Copies all tensor data to a FloatArray.
     *
     * This method provides efficient bulk data transfer from tensor storage to a FloatArray.
     * Backend implementations (e.g., GPU backends) can override this to provide optimized
     * bulk copy operations instead of element-by-element access.
     *
     * The default implementation iterates over all elements, which may be slow for backends
     * where individual element access is expensive (e.g., GPU tensors).
     *
     * The default unravels each flat position into per-dimension indices, because [get]
     * requires exactly one index per dimension — a single flat index would trip every
     * implementation's arity check for rank >= 2 tensors.
     *
     * @return a new FloatArray containing all tensor values in row-major order
     */
    public fun copyToFloatArray(): FloatArray {
        val dims = shape.dimensions
        val volume = shape.volume
        val indices = IntArray(dims.size)
        return FloatArray(volume) { flat ->
            var remaining = flat
            for (d in dims.indices.reversed()) {
                indices[d] = remaining % dims[d]
                remaining /= dims[d]
            }
            (get(*indices) as Number).toFloat()
        }
    }
}

/**
 * Marker interface for tensor data backed by a contiguous `FloatArray`.
 * Provides direct buffer access for performance-critical backends.
 */
public interface FloatArrayTensorData<T : DType> : TensorData<T, Float> {
    public val buffer: FloatArray

    override fun copyToFloatArray(): FloatArray = buffer.copyOf()

    /** A dense FP32 view borrowing [buffer] — zero-copy, the same bytes this data reads and writes. */
    override val view: sk.ainet.lang.memory.TensorView
        get() = sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.Storage.Heap.wrap(buffer),
            shape,
            sk.ainet.lang.types.FP32,
        )
}

/**
 * Marker interface for tensor data backed by a contiguous `IntArray`.
 */
public interface IntArrayTensorData<T : DType> : TensorData<T, Int> {
    public val buffer: IntArray

    /** A dense Int32 view borrowing [buffer] — zero-copy, the same bytes this data reads and writes. */
    override val view: sk.ainet.lang.memory.TensorView
        get() = sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.Storage.Heap.wrap(buffer),
            shape,
            sk.ainet.lang.types.Int32,
        )
}
