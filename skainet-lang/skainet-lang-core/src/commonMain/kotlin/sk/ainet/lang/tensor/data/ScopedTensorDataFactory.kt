package sk.ainet.lang.tensor.data

import sk.ainet.lang.memory.Scope
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * [base] with dense-FP32 allocations drawn from the active [Scope] — the op-output half of the
 * Scope split (#1146).
 *
 * The creation path (`ExecutionContext.zeros/…`) learned to consult `memoryScope` in #1145; op
 * *outputs* still reached the GC because `DefaultCpuOps` allocates them through its
 * [TensorDataFactory]. This decorator is that factory: when [scope] is anything other than
 * [Scope.Ambient] and the dtype is dense FP32, `zeros`/`ones`/`full`/`init`/`fromFloatArray`
 * return [StorageFloatTensorData] over the scope's slab — recycled at `ForwardScope.reset()`,
 * loud after it. Everything else falls through to [base] untouched.
 *
 * The slab is *not* zeroed between steps, so every intercepted method fully writes its region:
 * `zeros`/`ones`/`full` fill, `init` writes each element, `fromFloatArray` copies the source in.
 *
 * `wrapFloatArray`/`wrapIntArray`/`wrapByteArray` are deliberately **not** intercepted: their
 * contract is zero-copy over a caller-owned array (loaders use them for weights), and neither
 * copying them into a slab nor letting them die at `reset()` would honour it.
 */
public class ScopedTensorDataFactory(
    private val base: TensorDataFactory,
    private val scope: () -> Scope,
) : TensorDataFactory by base {

    /** The scope currently in effect — lets an ops implementation holding this factory pass the
     *  same scope to kernel dispatch, so adapter allocations land in the slab too (#1146). */
    public val currentScope: Scope get() = scope()

    private fun <T : DType> slab(shape: Shape, dtype: KClass<T>): StorageFloatTensorData<T>? {
        val s = scope()
        if (s === Scope.Ambient || dtype != FP32::class) return null
        return StorageFloatTensorData(shape, s.allocateFloats(shape.volume))
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> fill(scoped: StorageFloatTensorData<T>, value: Float): TensorData<T, V> {
        val st = scoped.storage
        st.floats!!.fill(value, st.arrayOffset, st.arrayOffset + scoped.shape.volume)
        return scoped as TensorData<T, V>
    }

    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V> =
        slab(shape, dtype)?.let { fill(it, 0f) } ?: base.zeros(shape, dtype)

    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): TensorData<T, V> =
        slab(shape, dtype)?.let { fill(it, 1f) } ?: base.ones(shape, dtype)

    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): TensorData<T, V> =
        slab(shape, dtype)?.let { fill(it, value.toFloat()) } ?: base.full(shape, dtype, value)

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromFloatArray(shape: Shape, dtype: KClass<T>, data: FloatArray): TensorData<T, V> {
        val scoped = slab(shape, dtype) ?: return base.fromFloatArray(shape, dtype, data)
        val st = scoped.storage
        data.copyInto(st.floats!!, st.arrayOffset, 0, shape.volume)
        return scoped as TensorData<T, V>
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> adoptFloatArray(shape: Shape, dtype: KClass<T>, data: FloatArray): TensorData<T, V> {
        val scoped = slab(shape, dtype) ?: return base.adoptFloatArray(shape, dtype, data)
        val st = scoped.storage
        data.copyInto(st.floats!!, st.arrayOffset, 0, shape.volume)
        return scoped as TensorData<T, V>
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> init(
        shape: Shape,
        dtype: KClass<T>,
        generator: (indices: IntArray) -> V,
    ): TensorData<T, V> {
        val scoped = slab(shape, dtype) ?: return base.init(shape, dtype, generator)
        val st = scoped.storage
        val floats = st.floats!!
        val dims = shape.dimensions
        val indices = IntArray(dims.size)
        val volume = shape.volume
        // Row-major walk, same visiting order as the dense factory.
        for (flat in 0 until volume) {
            var remaining = flat
            for (d in dims.indices.reversed()) {
                indices[d] = remaining % dims[d]
                remaining /= dims[d]
            }
            floats[st.arrayOffset + flat] = generator(indices) as Float
        }
        return scoped as TensorData<T, V>
    }
}
