package sk.ainet.context

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.operators.OpsBoundTensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.tensor.scratch.NoopScratchPool
import sk.ainet.lang.tensor.scratch.ScratchPool
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

public interface ExecutionContext {
    /**
     * Where this context's trace events go (SKEEP-003 §4.9): phases, kernel runs, adapter
     * insertions, allocations. Default [sk.ainet.lang.memory.trace.NoopTraceSink] — nothing is
     * recorded until a context opts in with a recording or exporting sink.
     */
    public val traceSink: sk.ainet.lang.memory.trace.TraceSink get() = sk.ainet.lang.memory.trace.NoopTraceSink

    /**
     * The scope new activations and adapter outputs are allocated in (SKEEP-003 §4.5). Default
     * [sk.ainet.lang.memory.Scope.Ambient] — GC-managed, today's behaviour; a generation loop opts
     * in by providing a `ForwardScope` and calling `reset()` per step.
     */
    public val memoryScope: sk.ainet.lang.memory.Scope get() = sk.ainet.lang.memory.Scope.Ambient

    public val ops: TensorOps

    // Optional forward hooks for recording or diagnostics (null → disabled)
    public val hooks: sk.ainet.lang.nn.hooks.ForwardHooks? get() = null

    // Execution phase and convenience training flag
    public val phase: Phase
    public val inTraining: Boolean get() = phase == Phase.TRAIN

    /**
     * Whether this context is currently *recording* a trace/graph (vs. plain eager execution).
     * Defaults to `false` (eager); the graph/tape context overrides it. Modules with an eager
     * fast-path that bypasses `ops.*` (e.g. RoPE's raw-array interleaved rotation) can check this
     * to emit a graph-traceable `ctx.ops.*` path instead when recording, so they export to
     * StableHLO while keeping the fast path for eager inference.
     */
    public val isRecording: Boolean get() = false

    public val tensorDataFactory: TensorDataFactory

    /**
     * This context rebuilt around [factory] — the seam that lets a decorator (notably
     * `ScopedExecutionContext`, #1146) swap in a scope-aware [TensorDataFactory] and get an
     * *ops instance that allocates through it*, since ops are constructed from the factory.
     *
     * The default returns `this` unchanged: a context that cannot rebuild itself simply keeps
     * its own factory, and op outputs stay GC-allocated (correct, just not scope-recycled).
     * Contexts that own their ops construction (e.g. `DirectCpuExecutionContext`) override.
     */
    public fun withTensorDataFactory(factory: TensorDataFactory): ExecutionContext = this

    /**
     * How this context's ops map the independent work inside one op onto cores (SKEEP-005).
     * Default [sk.ainet.context.schedule.Schedule.Sequential]: one task, inline, today's behaviour.
     * A schedule never changes a result — it is a deployment property, not a model property.
     */
    public val schedule: sk.ainet.context.schedule.Schedule
        get() = sk.ainet.context.schedule.Schedule.Sequential

    /**
     * This context rebuilt so its ops run under [schedule] — the same seam as
     * [withTensorDataFactory]. The default cannot rebuild, returns `this`, and — unless the request
     * is already what this context runs — emits [sk.ainet.lang.memory.trace.TraceEvent.ScheduleDowngraded]
     * so an unhonoured schedule is visible in the trace instead of silently sequential.
     */
    public fun withSchedule(schedule: sk.ainet.context.schedule.Schedule): ExecutionContext {
        if (schedule !== this.schedule && traceSink.isEnabled) {
            traceSink.emit(
                sk.ainet.lang.memory.trace.TraceEvent.ScheduleDowngraded(
                    requested = schedule.name,
                    effective = this.schedule.name,
                    reason = "context ${this::class.simpleName} cannot rebuild its ops",
                ),
            )
        }
        return this
    }

    /**
     * Workspace allocator for short-lived intermediate buffers (attention
     * scratch, RoPE tables, KV-cache slice copies, padding scratch, etc.).
     *
     * Default is [NoopScratchPool] — every acquire allocates a fresh array,
     * matching pre-pool behavior. Implementations that want pooling override
     * this property (or wrap an existing context).
     *
     * Callers MUST acquire inside an active [ScratchPool.scope] block;
     * acquires outside a scope succeed but the buffer is not returned to the
     * pool when dropped.
     *
     * Boundary with [memoryScope], on purpose (#1135): `scratch` is untyped *intra-kernel*
     * workspace — raw arrays inside one op invocation, returned when its block exits.
     * [memoryScope] governs *inter-op* activation lifetime — typed tensors that live across ops
     * within a step and are recycled by `ForwardScope.reset()`. They are different layers and
     * stay separate; neither replaces the other.
     */
    public val scratch: ScratchPool get() = NoopScratchPool

    // Execution observers for tracing/benchmarking
    public val observers: ExecutionObserverRegistry

    public fun registerObserver(observer: ExecutionObserver) {
        observers.register(observer)
    }

    public fun unregisterObserver(observer: ExecutionObserver) {
        observers.unregister(observer)
    }

    /**
     * Dense FP32 data drawn from [memoryScope] when a scope other than `Ambient` is active — the
     * creation-path reader of the Scope split (#1145). `null` on the Ambient default, so the
     * factory path is untouched for every context that never opts in. The region is *not* cleared:
     * a slab slice after `reset()` holds old bytes, so callers fill it themselves.
     */
    private fun <T : DType> scopedDenseFloats(
        shape: Shape,
        dtype: KClass<T>,
    ): sk.ainet.lang.tensor.data.StorageFloatTensorData<T>? {
        val scope = memoryScope
        if (scope === sk.ainet.lang.memory.Scope.Ambient || dtype != sk.ainet.lang.types.FP32::class) return null
        return sk.ainet.lang.tensor.data.StorageFloatTensorData(shape, scope.allocateFloats(shape.volume))
    }

    public fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): Tensor<T, V> {
        scopedDenseFloats(shape, dtype)?.let { scoped ->
            val s = scoped.storage
            s.floats!!.fill(value.toFloat(), s.arrayOffset, s.arrayOffset + shape.volume)
            @Suppress("UNCHECKED_CAST")
            return fromData(scoped as TensorData<T, V>, dtype)
        }
        val data = tensorDataFactory.full<T, V>(shape, dtype, value)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> zeros(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        scopedDenseFloats(shape, dtype)?.let { scoped ->
            val s = scoped.storage
            s.floats!!.fill(0f, s.arrayOffset, s.arrayOffset + shape.volume)
            @Suppress("UNCHECKED_CAST")
            return fromData(scoped as TensorData<T, V>, dtype)
        }
        val data = tensorDataFactory.zeros<T, V>(shape, dtype)
        return fromData(data, dtype)
    }

    /**
     * Lazy-initialized zero tensor — see [TensorDataFactory.placeholder].
     * The underlying primitive array allocates on first read; if the parameter
     * is replaced before any read (the common case for DSL modules whose weights
     * are loaded from disk), the allocation is skipped entirely.
     */
    public fun <T : DType, V> placeholder(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        val data = tensorDataFactory.placeholder<T, V>(shape, dtype)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> ones(
        shape: Shape,
        dtype: KClass<T>
    ): Tensor<T, V> {
        if (memoryScope !== sk.ainet.lang.memory.Scope.Ambient) return full(shape, dtype, 1)
        val data = tensorDataFactory.ones<T, V>(shape, dtype)
        return fromData(data, dtype)
    }


    public fun <T : DType, V> fromData(data: TensorData<T, V>, dtype: KClass<T>): Tensor<T, V> = OpsBoundTensor.fromData(
        data, dtype,
        ops
    )

    public fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): Tensor<T, V> {
        scopedDenseFloats(shape, dtype)?.let { scoped ->
            val s = scoped.storage
            data.copyInto(s.floats!!, s.arrayOffset, 0, shape.volume)
            @Suppress("UNCHECKED_CAST")
            return fromData(scoped as TensorData<T, V>, dtype)
        }
        val data = tensorDataFactory.fromFloatArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromIntArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    public fun <T : DType, V> fromByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): Tensor<T, V> {
        val data = tensorDataFactory.fromByteArray<T, V>(shape, dtype, data)
        return fromData(data, dtype)
    }

    /**
     * Wraps a FloatArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapFloatArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    /**
     * Wraps an IntArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapIntArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    /**
     * Wraps a ByteArray without copying (borrow semantics).
     * The caller must ensure the array is not mutated while the tensor is in use.
     */
    public fun <T : DType, V> wrapByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray
    ): Tensor<T, V> {
        val tensorData = tensorDataFactory.wrapByteArray<T, V>(shape, dtype, data)
        return fromData(tensorData, dtype)
    }

    // runtime information
    public val memoryInfo: MemoryInfo
    public val executionStats: ExecutionStats

    /** Memory tracker for observability and copy tracing. Default: no-op (not tracking). */
    public val memoryTracker: MemoryTracker? get() = null
}
