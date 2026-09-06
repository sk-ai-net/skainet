package sk.ainet.context

import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * [base] with a [memoryScope] — the wiring #1135 asked about (#1145).
 *
 * The mechanism existed before this class did: [ForwardScope] bump-allocates a pre-sized slab and
 * `reset()` recycles it per step; what was missing was any reader of `ExecutionContext.memoryScope`.
 * This decorator is that reader's other half: creation methods on [ExecutionContext] consult
 * `memoryScope` when it is not [Scope.Ambient], so `zeros`/`full`/`ones`/`fromFloatArray` draw
 * from the scope's slab and their tensors die at `reset()`.
 *
 * `Ambient` remains the default everywhere — nothing changes for a context that never opts in.
 * The boundary with [ExecutionContext.scratch] is deliberate: `scratch` is untyped *intra-kernel*
 * workspace inside one op invocation; `memoryScope` governs *inter-op* activation lifetime across
 * a step. They stay separate.
 */
public class ScopedExecutionContext(
    private val base: ExecutionContext,
    override val memoryScope: Scope,
) : ExecutionContext by base {

    /**
     * The base rebuilt around a scope-aware factory, so *op outputs* allocate from the slab too
     * (#1146). A base that cannot rebuild itself (default [ExecutionContext.withTensorDataFactory])
     * returns itself — creation still draws from the scope, op outputs stay GC-allocated.
     *
     * Note the ops-binding rule: `a + b` dispatches through the ops instance that *created* `a`,
     * so only tensors created through this context (or explicitly re-bound) produce scoped
     * outputs. Tensors made before entering the scope keep their unscoped ops — deliberately, so
     * their results do not die at `reset()`.
     */
    private val scopedBase: ExecutionContext =
        base.withTensorDataFactory(
            sk.ainet.lang.tensor.data.ScopedTensorDataFactory(base.tensorDataFactory) { memoryScope },
        )

    override val tensorDataFactory: sk.ainet.lang.tensor.data.TensorDataFactory
        get() = scopedBase.tensorDataFactory

    override val ops: sk.ainet.lang.tensor.ops.TensorOps
        get() = scopedBase.ops

    /**
     * Bind tensors made through this context to the *scoped* ops — `a + b` dispatches through
     * the ops that created `a`, and only the scoped ops allocate outputs from the slab.
     */
    override fun <T : DType, V> fromData(
        data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        dtype: KClass<T>,
    ): Tensor<T, V> = sk.ainet.lang.tensor.operators.OpsBoundTensor.fromData(data, dtype, ops)

    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): Tensor<T, V> =
        super.zeros(shape, dtype)

    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): Tensor<T, V> =
        super.ones(shape, dtype)

    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): Tensor<T, V> =
        super.full(shape, dtype, value)

    override fun <T : DType, V> fromFloatArray(shape: Shape, dtype: KClass<T>, data: FloatArray): Tensor<T, V> =
        super.fromFloatArray(shape, dtype, data)
}

/**
 * Run [block] with a [ForwardScope] of [slabFloats] floats active on this context, closing the
 * scope (and everything it handed out) afterwards. Call `reset()` on the scope between steps:
 *
 * ```kotlin
 * ctx.forwardScope(slabFloats = 1 shl 20) { scoped, scope ->
 *     while (decoding) {
 *         step(scoped)
 *         scope.reset()   // steady state: zero new slab bytes per step
 *     }
 * }
 * ```
 */
public inline fun <R> ExecutionContext.forwardScope(
    slabFloats: Int,
    block: (ctx: ScopedExecutionContext, scope: ForwardScope) -> R,
): R {
    val scope = ForwardScope(slabFloats, traceSink)
    return scope.use { block(ScopedExecutionContext(this, scope), scope) }
}
