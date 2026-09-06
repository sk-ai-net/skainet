package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.ops.Backend
import sk.ainet.lang.ops.TensorOp
import sk.ainet.lang.ops.InProgress
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.lang.tensor.data.RowDequantSource
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.IntArrayTensorData
import sk.ainet.lang.tensor.data.NarrowFloatInputMajorTensorData
import sk.ainet.lang.tensor.data.TransposedWeightTensorData
import sk.ainet.lang.tensor.data.Q4_0TensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_KTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_1TensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_0TensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10 as kmLog10
import kotlin.math.log2 as kmLog2
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.reflect.KClass

/**
 * Below this many multiply-adds a scaled-dot-product-attention call runs on the caller's thread
 * regardless of the context's schedule: a coroutine region costs more than it saves (SKEEP-005).
 * 8 heads × 64 keys × 128 dims × 2 ≈ 131k is well under it; a 512-token prefill chunk is far over.
 */
internal const val SDPA_PARALLEL_MIN_WORK: Long = 1L shl 20

@Backend(id = "cpu", displayName = "CPU")
@InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#defaultcpuops")
public open class DefaultCpuOpsBase(
    protected val dataFactory: TensorDataFactory,
    /** How this ops instance maps independent work onto cores (SKEEP-005); [Schedule.Sequential] by default. */
    protected val schedule: sk.ainet.context.schedule.Schedule,
) : TensorOps {
    public constructor(dataFactory: TensorDataFactory) : this(dataFactory, sk.ainet.context.schedule.Schedule.Sequential)


    protected class CpuTensor<T : DType, V>(
        override val data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        private val opsRef: TensorOps,
        override val dtype: kotlin.reflect.KClass<T>,
        override val gradState: GradState<T, V> = GradState()
    ) : Tensor<T, V> {
        override val ops: TensorOps
            get() = opsRef
    }

    protected fun <T : DType, V> gradStateFrom(vararg tensors: Tensor<T, V>): GradState<T, V> {
        val requires = tensors.any { it.requiresGrad }
        return GradState(requiresGrad = requires)
    }

    protected fun <T : DType, V> newTensor(
        data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        dtype: kotlin.reflect.KClass<T>,
        vararg inputs: Tensor<T, V>
    ): Tensor<T, V> = CpuTensor(data, this, dtype, gradStateFrom(*inputs))

    private fun rowMajorStrides(shape: Shape): IntArray {
        val strides = IntArray(shape.rank)
        var stride = 1
        for (i in shape.rank - 1 downTo 0) {
            strides[i] = stride
            stride *= shape[i]
        }
        return strides
    }

    private fun flatIndexToIndices(flatIndex: Int, strides: IntArray): IntArray {
        val indices = IntArray(strides.size)
        var remaining = flatIndex
        for (i in strides.indices) {
            indices[i] = remaining / strides[i]
            remaining %= strides[i]
        }
        return indices
    }

    private fun <T : DType, V> copyTensorValuesAsFloatArray(tensor: Tensor<T, V>): FloatArray {
        val data = tensor.data
        return when (data) {
            is FloatArrayTensorData<*> -> data.buffer.copyOf()
            is IntArrayTensorData<*> -> FloatArray(data.buffer.size) { data.buffer[it].toFloat() }
            else -> {
                val strides = rowMajorStrides(tensor.shape)
                FloatArray(tensor.shape.volume) { flatIndex ->
                    val indices = flatIndexToIndices(flatIndex, strides)
                    (data.get(*indices) as Number).toFloat()
                }
            }
        }
    }

    private fun <T : DType, V> copyTensorValuesAsIntArray(tensor: Tensor<T, V>): IntArray {
        val data = tensor.data
        return when (data) {
            is IntArrayTensorData<*> -> data.buffer.copyOf()
            is FloatArrayTensorData<*> -> IntArray(data.buffer.size) { data.buffer[it].toInt() }
            else -> {
                val strides = rowMajorStrides(tensor.shape)
                IntArray(tensor.shape.volume) { flatIndex ->
                    val indices = flatIndexToIndices(flatIndex, strides)
                    (data.get(*indices) as Number).toInt()
                }
            }
        }
    }

    protected fun broadcastShapes(a: Shape, b: Shape): Shape {
        val ad = a.dimensions
        val bd = b.dimensions
        val maxRank = maxOf(ad.size, bd.size)
        val out = IntArray(maxRank)
        var ai = ad.size - 1
        var bi = bd.size - 1
        for (oi in maxRank - 1 downTo 0) {
            val asz = if (ai >= 0) ad[ai] else 1
            val bsz = if (bi >= 0) bd[bi] else 1
            if (asz != bsz && asz != 1 && bsz != 1) {
                throw IllegalArgumentException("Shapes ${a.dimensions.contentToString()} and ${b.dimensions.contentToString()} cannot be broadcasted")
            }
            out[oi] = maxOf(asz, bsz)
            ai--; bi--
        }
        return Shape(out)
    }

    protected fun mapIndex(idx: IntArray, inShape: Shape): IntArray {
        // Map output index to input index with broadcasting: if input dim == 1, use 0 for that dim.
        val inDims = inShape.dimensions
        val outRank = idx.size
        val inRank = inDims.size
        val mapped = IntArray(inRank)
        var ir = inRank - 1
        var or = outRank - 1
        while (ir >= 0) {
            val inDim = inDims[ir]
            val outIndex = if (or >= 0) idx[or] else 0
            mapped[ir] = if (inDim == 1) 0 else outIndex
            ir--; or--
        }
        return mapped
    }

    protected fun <T : DType, V> requireSameDType(a: Tensor<T, V>, b: Tensor<T, V>) {
        require(a.dtype == b.dtype) { "DType mismatch: ${'$'}{a.dtype} vs ${'$'}{b.dtype}" }
    }

    protected fun <T : DType, V> elementwise(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        op: (av: V, bv: V, dtype: kotlin.reflect.KClass<T>) -> V
    ): Tensor<T, V> {
        requireSameDType(a, b)
        val outShape = broadcastShapes(a.shape, b.shape)
        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            val ai = mapIndex(outIdx, a.shape)
            val bi = mapIndex(outIdx, b.shape)
            val av = a.data.get(*ai)
            val bv = b.data.get(*bi)
            op(av, bv, a.dtype)
        }
        return newTensor(outData, a.dtype, a, b)
    }

    // ---- FP32 primitive fast paths (#949) --------------------------------
    //
    // The generic paths in this class pay, per element: IntArray allocations
    // for broadcast index mapping, a vararg-spread boxed `data.get`, a KClass
    // `when (dtype)` comparison, and a boxed lambda round-trip. On ART that
    // overhead dominates LLM decode (#949: 83% of e2e decode on a Pixel 8a
    // was non-matmul overhead, most of it these mechanics). The helpers below
    // give the hot ops a flat primitive loop over the dense FloatArray buffer;
    // every caller falls back to the generic path for other dtypes/layouts.
    // `inline` is load-bearing: a non-inlined `(Float, Float) -> Float` lambda
    // would box through `Function2` and reintroduce the very churn removed.

    private fun <T : DType, V> floatBufferOf(t: Tensor<T, V>): FloatArray? = when (val d = t.data) {
        is FloatArrayTensorData<*> -> d.buffer
        // Slab-backed data (#1145/#1146) has a nonzero base offset, so it cannot hand out its raw
        // array — but its exact logical window as one copy still beats the boxed generic path by
        // orders of magnitude (#949). The zero-copy path for the hot trio is [floatWindowOf].
        is sk.ainet.lang.tensor.data.StorageFloatTensorData<*> -> d.copyToFloatArray()
        else -> null
    }

    /** Zero-copy dense-FP32 window: the backing array plus the base offset of element 0 (#1146). */
    private class FloatWindow(val arr: FloatArray, val off: Int)

    private fun <T : DType, V> floatWindowOf(t: Tensor<T, V>): FloatWindow? = when (val d = t.data) {
        is FloatArrayTensorData<*> -> FloatWindow(d.buffer, 0)
        is sk.ainet.lang.tensor.data.StorageFloatTensorData<*> -> {
            val s = d.storage
            s.checkAlive()
            FloatWindow(s.floats!!, s.arrayOffset)
        }
        else -> null
    }

    /**
     * The scope the factory is placing outputs in, or `Ambient` — passed to kernel dispatch so
     * adapter allocations (requantized activations, prepacked weights) land in the slab too (#1146).
     */
    protected fun dispatchScope(): sk.ainet.lang.memory.Scope =
        (dataFactory as? sk.ainet.lang.tensor.data.ScopedTensorDataFactory)?.currentScope
            ?: sk.ainet.lang.memory.Scope.Ambient

    /**
     * A freshly computed dense-FP32 result, adopted through the factory — the single funnel a
     * scope-aware factory (#1146) intercepts to place op outputs in the active scope. [buf] is
     * ops-owned and never touched again; the default factory adopts it zero-copy.
     */
    protected fun <T : DType, V> floatResult(
        shape: Shape,
        dtype: kotlin.reflect.KClass<T>,
        buf: FloatArray,
        vararg inputs: Tensor<T, V>,
    ): Tensor<T, V> {
        @Suppress("UNCHECKED_CAST")
        val outData = dataFactory.adoptFloatArray<T, Float>(shape, dtype, buf)
            as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(outData, dtype, *inputs)
    }

    private inline fun <T : DType, V> floatUnaryFast(
        t: Tensor<T, V>,
        op: (Float) -> Float,
    ): Tensor<T, V>? {
        val src = floatWindowOf(t) ?: return null
        val n = t.shape.volume
        val sa = src.arr
        val so = src.off
        val out = FloatArray(n)
        for (i in 0 until n) out[i] = op(sa[so + i])
        return floatResult(t.shape, t.dtype, out, t)
    }

    private inline fun <T : DType, V> floatBinaryFast(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        op: (Float, Float) -> Float,
    ): Tensor<T, V>? {
        if (a.dtype != b.dtype) return null
        val aw = floatWindowOf(a) ?: return null
        val bw = floatWindowOf(b) ?: return null
        val ab = aw.arr
        val ao = aw.off
        val bb = bw.arr
        val bo = bw.off
        val outShape = try {
            broadcastShapes(a.shape, b.shape)
        } catch (e: IllegalArgumentException) {
            return null
        }
        val n = outShape.volume
        if (a.shape == b.shape) {
            val out = FloatArray(n)
            for (i in 0 until n) out[i] = op(ab[ao + i], bb[bo + i])
            return floatResult(outShape, a.dtype, out, a, b)
        }
        if (a.shape.volume == 1) {
            val av = ab[ao]
            val out = FloatArray(n)
            for (i in 0 until n) out[i] = op(av, bb[bo + i])
            return floatResult(outShape, a.dtype, out, a, b)
        }
        if (b.shape.volume == 1) {
            val bv = bb[bo]
            val out = FloatArray(n)
            for (i in 0 until n) out[i] = op(ab[ao + i], bv)
            return floatResult(outShape, a.dtype, out, a, b)
        }
        // Last-dim ("bias") broadcast, mirroring DefaultCpuOpsJvm.vectorFloatBinary.
        val outLast = outShape.dimensions.lastOrNull() ?: return null
        if (outLast <= 0) return null
        val groups = n / outLast
        val bIsBias = b.shape.rank >= 1 && b.shape[b.shape.rank - 1] == outLast &&
            b.shape.dimensions.dropLast(1).all { it == 1 }
        val aIsBias = a.shape.rank >= 1 && a.shape[a.shape.rank - 1] == outLast &&
            a.shape.dimensions.dropLast(1).all { it == 1 }
        if (bIsBias && a.shape.volume == n) {
            val out = FloatArray(n)
            for (g in 0 until groups) {
                val off = g * outLast
                for (i in 0 until outLast) out[off + i] = op(ab[ao + off + i], bb[bo + i])
            }
            return floatResult(outShape, a.dtype, out, a, b)
        }
        if (aIsBias && b.shape.volume == n) {
            val out = FloatArray(n)
            for (g in 0 until groups) {
                val off = g * outLast
                for (i in 0 until outLast) out[off + i] = op(ab[ao + i], bb[bo + off + i])
            }
            return floatResult(outShape, a.dtype, out, a, b)
        }
        return null
    }

    // Scalar ops implemented via materializing a full-like tensor and delegating to elementwise ops
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val s = b.toFloat()
        floatUnaryFast(a) { x -> x + s }?.let { return it }
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return add(a, sb)
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val s = b.toFloat()
        floatUnaryFast(a) { x -> x - s }?.let { return it }
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return subtract(a, sb)
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val s = b.toFloat()
        floatUnaryFast(a) { x -> x * s }?.let { return it }
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return multiply(a, sb)
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val s = b.toFloat()
        floatUnaryFast(a) { x -> x / s }?.let { return it }
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return divide(a, sb)
    }

    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val s = a.toFloat()
        floatUnaryFast(b) { x -> s - x }?.let { return it }
        val ta = newTensor(
            dataFactory.full<T, V>(b.shape, b.dtype, a),
            b.dtype,
            b
        )
        return subtract(ta, b)
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val s = a.toFloat()
        floatUnaryFast(b) { x -> s / x }?.let { return it }
        val ta = newTensor(
            dataFactory.full<T, V>(b.shape, b.dtype, a),
            b.dtype,
            b
        )
        return divide(ta, b)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-add")
    override fun <T : DType, V> add(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        floatBinaryFast(a, b) { x, y -> x + y }?.let { return it }
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x + y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x + y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for add: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-subtract")
    override fun <T : DType, V> subtract(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        floatBinaryFast(a, b) { x, y -> x - y }?.let { return it }
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x - y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x - y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for subtract: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-multiply")
    override fun <T : DType, V> multiply(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        floatBinaryFast(a, b) { x, y -> x * y }?.let { return it }
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x * y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x * y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for multiply: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-divide")
    override fun <T : DType, V> divide(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        floatBinaryFast(a, b) { x, y -> x / y }?.let { return it }
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x / y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; if (y == 0) 0 as V else (x / y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for divide: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    /**
     * Hook to populate [KernelRegistry] before the platform-neutral packed-quant
     * dispatch resolves kernels. No-op in the base (callers register providers
     * directly, e.g. the non-JVM platform factories register [ScalarKernelProvider]);
     * the JVM ops override this to auto-install ServiceLoader-discovered providers.
     */
    protected open fun ensureKernelProviders() {}

    private fun resolveProvider(test: (KernelProvider) -> Boolean): KernelProvider? {
        ensureKernelProviders()
        return KernelRegistry.providers().firstOrNull { it.isAvailable() && test(it) }
    }

    private val q8_0Kernel by lazy { resolveProvider { it.matmulQ8_0() != null }?.matmulQ8_0() }
    private val q4_0Kernel by lazy { resolveProvider { it.matmulQ4_0() != null }?.matmulQ4_0() }
    private val q4kKernel by lazy { resolveProvider { it.matmulQ4K() != null }?.matmulQ4K() }
    private val q6kKernel by lazy { resolveProvider { it.matmulQ6K() != null }?.matmulQ6K() }
    private val q5kKernel by lazy { resolveProvider { it.matmulQ5K() != null }?.matmulQ5K() }
    private val q5_1Kernel by lazy { resolveProvider { it.matmulQ5_1() != null }?.matmulQ5_1() }
    private val q5_0Kernel by lazy { resolveProvider { it.matmulQ5_0() != null }?.matmulQ5_0() }

    /**
     * `cols / blockSize`, i.e. the number of quantization blocks per output row —
     * validated so a misaligned packed tensor fails loudly here rather than
     * silently truncating a partial trailing block during [transposePackedBlocks].
     */
    private fun requirePackedBlockAligned(cols: Int, blockSize: Int, formatName: String): Int {
        require(cols % blockSize == 0) {
            "$formatName transpose: inputDim $cols is not a multiple of the block size $blockSize " +
                "— packed weight is not row-block-aligned, cannot compute a physical block-grid transpose"
        }
        return cols / blockSize
    }

    /**
     * Physically transposes a packed weight's block grid from *row-major*
     * (canonical) order into the *input-block-major* order the packed-quant
     * matmul kernels require.
     *
     * Canonical storage — what a fresh `[outputDim, inputDim]`-shaped packed
     * tensor has, whether loaded verbatim from GGUF or built any other
     * row-major way — groups blocks per output row: for row `o`, its
     * `blocksPerInputDim` blocks are contiguous, then row `o + 1`. Flat block
     * index = `o * blocksPerInputDim + blockIdx`.
     *
     * Every packed-quant native/Panama/scalar matmul kernel instead reads
     * `weight + (blockIdx * outputDim + o) * bytesPerBlock` — see e.g. the
     * `Per-block packed weight layout` header comment in `q5_0_matmul.c` /
     * `q4k_matmul.c` / etc. — because for a FIXED input block, all `outputDim`
     * rows' corresponding block bytes are consecutive, letting the kernel's
     * block-outer/row-inner loop read weight memory sequentially instead of
     * striding `outputDim * bytesPerBlock` per input block.
     *
     * These two orderings are literal transposes of the `(outputDim,
     * blocksPerInputDim)` block grid (treating each `bytesPerBlock`-sized
     * block as an atomic item) and coincide only when `blocksPerInputDim == 1`.
     * For every wider weight — i.e. essentially every real model, since
     * `inputDim` is almost always many multiples of the 32/256-element block
     * size — reordering the *shape* without reordering the *bytes* hands the
     * kernel physically wrong data: it silently reads block `bI` of row `o`
     * from where block `o`'s row `bI`-th chunk actually lives. This is the
     * root cause of the SKaiNET-transformers#307 all-zero Q5_0/Q5_1 matmul
     * report (general to every packed format the native tier serves, not
     * Q5-specific — see `NativeLazyTransposeGroundTruthReproTest`).
     */
    private fun transposePackedBlocks(
        packed: ByteArray,
        outputDim: Int,
        blocksPerInputDim: Int,
        bytesPerBlock: Int,
    ): ByteArray {
        val out = ByteArray(packed.size)
        for (o in 0 until outputDim) {
            for (blockIdx in 0 until blocksPerInputDim) {
                val srcOff = (o * blocksPerInputDim + blockIdx) * bytesPerBlock
                val dstOff = (blockIdx * outputDim + o) * bytesPerBlock
                packed.copyInto(out, dstOff, srcOff, srcOff + bytesPerBlock)
            }
        }
        return out
    }

    /**
     * Platform-neutral packed-quant matmul: `FP32 input × packed-quant weight`,
     * resolving the kernel via [KernelRegistry] (scalar on Native/JS/WASM, Panama/
     * native-FFM on JVM). Returns `null` when the weight isn't a heap-packed quant
     * type or no provider carries a kernel, so callers fall through. The JVM ops
     * intercept Q4_K/Q6_K/Q8_0/Q4_0 (+ MemSeg) before this runs; Q5_1/Q5_0 (and the
     * whole set on non-JVM) resolve here.
     */
    protected fun <T : DType, V> chooseQuantizedMatmulHeap(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (a.dtype != FP32::class || b.shape.rank != 2 || a.shape.rank < 1) return null
        if (a.shape.rank == 2) return chooseQuantizedMatmulHeap2D(a, b)

        // Attention linear projections legitimately pass `[..., in]` (see linearProject's kdoc) —
        // flatten the leading batch/sequence dims into one so the specialized quant kernels below
        // (which only understand `[batch, in]`) still get used, instead of silently falling
        // through to matmulGeneric, which has no packed-quant handling at all (see SKaiNET#991).
        // Also covers rank-1 activations (a single-token hidden-state vector during incremental
        // decode, once the KV cache is warm and a matmul no longer runs against a batched
        // prefill) — `leading` is then empty and `flatBatch` is 1, i.e. `[in]` promotes to
        // `[1, in]` and the result squeezes back down to `[out]`, the same as `rank > 2`
        // already did; previously this rank guard sent every post-prefill decode step straight
        // to matmulGeneric's untyped per-element `TensorData.get()` path, which — for a
        // packed-quant (or pre-transposed-marker-wrapped, e.g. PreTransposedQ4_K) weight —
        // returns the raw packed byte, not a dequantized Float.
        val leading = a.shape.dimensions.copyOf(a.shape.rank - 1)
        val flatBatch = leading.fold(1) { acc, d -> acc * d }
        val inputDim = a.shape.dimensions.last()
        val a2d = reshape(a, Shape(intArrayOf(flatBatch, inputDim)))
        val result2d = chooseQuantizedMatmulHeap2D(a2d, b) ?: return null
        val outputDim = result2d.shape.dimensions.last()
        return reshape(result2d, Shape(leading + outputDim))
    }

    private fun <T : DType, V> chooseQuantizedMatmulHeap2D(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (a.shape[1] != b.shape[0]) return null
        // Any TensorData exposes copyToFloatArray() (FloatArrayTensorData overrides it with a cheap
        // buffer.copyOf(); everything else — e.g. MemorySegmentTensorData — uses the generic
        // row-major default). The previous strict `as? FloatArrayTensorData` cast meant activations
        // backed by anything else (e.g. MemorySegment-backed FP32, as SKaiNET-transformers' attention
        // path produces) silently declined here, falling through to the unguarded matmulGeneric
        // fallback for quant types routed to this function (Q6_K, Q5_1, Q5_0) — see SKaiNET#991.
        val inputBuffer = a.data.copyToFloatArray()
        val batchSize = a.shape[0]
        val inputDim = a.shape[1]
        val outputDim = b.shape[1]

        fun run(packed: ByteArray, kernel: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit): Tensor<T, V> {
            val out = FloatArray(batchSize * outputDim)
            for (batch in 0 until batchSize) {
                val bi = if (batchSize == 1) inputBuffer else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                kernel(bi, 0, packed, 0, inputDim, outputDim, out, batch * outputDim)
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.adoptFloatArray<T, Float>(Shape(batchSize, outputDim), a.dtype, out) as TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        return when (val bd = b.data) {
            is Q5_1TensorData -> q5_1Kernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q5_0TensorData -> q5_0Kernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q4_KTensorData -> q4kKernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q5_KTensorData -> q5kKernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q6_KTensorData -> q6kKernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q8_0TensorData -> q8_0Kernel?.let { k -> run(bd.packedData, k::matmul) }
            is Q4_0TensorData -> q4_0Kernel?.let { k -> run(bd.packedData, k::matmul) }
            else -> null
        }
    }

    override fun <T : DType, V> matmul(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        require(a.rank >= 1 && b.rank >= 1) { "Matrix multiplication requires tensors with at least 1 dimension per operand" }
        require(a.dtype == b.dtype) { "DType mismatch: ${a.dtype} vs ${b.dtype}" }

        // `x · Wᵀ` written as two steps (#1108). Must be first: below this line the weight would be
        // probed with `is PackedBlockStorage` and friends, and the marker deliberately answers no
        // to all of them, so it would fall through to a dense path and read a packed payload as
        // floats. Routing it here is what makes `x.matmul(w.t())` mean the same thing whatever the
        // weight is stored as.
        untransposedWeight(b)?.let { return matmulWeightTransposed(a, it) }

        // Packed-quant fast path (FP32 input × packed weight), resolved via KernelRegistry.
        KernelProfile.timeQuant { chooseQuantizedMatmulHeap(a, b) }?.let { return it }

        // Fast path: 2D × 2D with dense FP32 backing (array or slab window, #1146) — direct
        // buffer access, no per-element allocation
        if (a.rank == 2 && b.rank == 2 && (a.dtype == FP32::class)) {
            val aWin = floatWindowOf(a)
            val bWin = floatWindowOf(b)
            if (aWin != null && bWin != null) {
                return KernelProfile.timeFp32 {
                    val aBuf = aWin.arr
                    val aBase = aWin.off
                    val bBuf = bWin.arr
                    val bBase = bWin.off
                    val m = a.shape[0]
                    val k = a.shape[1]
                    val n = b.shape[1]
                    require(k == b.shape[0]) { "Matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}" }
                    val out = FloatArray(m * n)
                    // Loop order i-p-j, not i-j-p. The inner statement below walks `b` and `out`
                    // contiguously; the j-inner-p form it replaces read `b` with stride n, touching
                    // a fresh cache line on nearly every multiply — for a [1536, 256] weight that is
                    // 1536 lines per output element. Each output still accumulates its products in
                    // ascending p, so the result is bit-identical, not merely close.
                    for (i in 0 until m) {
                        val aOff = aBase + i * k
                        val outOff = i * n
                        for (p in 0 until k) {
                            val av = aBuf[aOff + p]
                            if (av == 0f) continue
                            val bOff = bBase + p * n
                            for (j in 0 until n) {
                                out[outOff + j] += av * bBuf[bOff + j]
                            }
                        }
                    }
                    @Suppress("UNCHECKED_CAST")
                    val outData = dataFactory.adoptFloatArray<T, Float>(Shape(m, n), a.dtype, out) as sk.ainet.lang.tensor.data.TensorData<T, V>
                    newTensor(outData, a.dtype, a, b)
                }
            }
        }

        // Everything else: the kernel registry first (SKEEP-003 §5.1) — rank is normalised once as
        // views and packed operands are read through decoding get(), so a rank-1 decode step against
        // a packed weight is correct by construction rather than a ClassCastException (#993). The
        // legacy per-element fallback stays one flag away (skainet.dispatch.registry=false) until the
        // migration is complete.
        return KernelProfile.timeGeneric {
            dispatchMatmulViaRegistry(a, b) ?: matmulGeneric(a, b)
        }
    }

    /**
     * Run `matmul` through [KernelDispatch] when both operands can describe themselves as views and
     * the result is float-typed; `null` means "not expressible here, use the legacy path".
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> dispatchMatmulViaRegistry(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!sk.ainet.backend.api.kernel.DispatchMode.useRegistry()) return null
        if (a.dtype != FP32::class) return null
        if (b.shape.rank != 2) return null
        val aView = a.data.view ?: return null
        val bView = b.data.view ?: return null
        // b is [k, n] here; the kernels take the weight output-major, which is a transposed *view*.
        val bT = try { bView.transpose() } catch (_: IllegalArgumentException) { return null }
        val (aNorm, leading) = try {
            sk.ainet.backend.api.kernel.KernelDispatch.normalizeActivation(aView)
        } catch (_: IllegalArgumentException) {
            return null
        }
        val m = aNorm.shape[0]
        val k = aNorm.shape[1]
        val n = bT.shape[0]
        if (k != bT.shape[1]) return null
        val outArray = FloatArray(m * n)
        val outView = sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.Storage.Heap.wrap(outArray),
            Shape(m, n),
            FP32,
        )
        sk.ainet.backend.api.kernel.KernelDispatch.matmul(aNorm, bT, outView, dispatchScope())
        val outShape = when {
            a.shape.rank == 1 -> Shape(n)                    // [k] x [k, n] -> [n]
            leading.isEmpty() -> Shape(m, n)
            else -> Shape(*(leading + n))
        }
        val outData = dataFactory.adoptFloatArray<T, Float>(outShape, a.dtype, outArray) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(outData, a.dtype, a, b)
    }

    private fun <T : DType, V> matmulGeneric(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        val aDims = a.shape.dimensions
        val bDims = b.shape.dimensions
        val aRank = aDims.size
        val bRank = bDims.size
        val aIs1D = aRank == 1
        val bIs1D = bRank == 1

        // Effective shapes (virtually unsqueeze 1D operands):
        val aEff = if (aIs1D) intArrayOf(1, aDims[0]) else aDims
        val bEff = if (bIs1D) intArrayOf(bDims[0], 1) else bDims
        val aEffRank = aEff.size
        val bEffRank = bEff.size

        val kA = aEff[aEffRank - 1]
        val kB = bEff[bEffRank - 2]
        require(kA == kB) { "Matrix multiplication shape mismatch: inner dimensions must match ($kA vs $kB)" }

        // Validate batch dims broadcastability on effective shapes (excluding last two dims)
        val maxEffRank = maxOf(aEffRank, bEffRank)
        for (i in 0 until maxEffRank - 2) {
            val aDim = if (i < aEffRank - 2) aEff[i] else 1
            val bDim = if (i < bEffRank - 2) bEff[i] else 1
            if (aDim != bDim && aDim != 1 && bDim != 1) {
                throw IllegalArgumentException("Matrix multiplication batch dimension mismatch at position $i: $aDim vs $bDim")
            }
        }

        // Compute output shape according to PyTorch rules (squeeze for 1D operands)
        val batchRank = maxEffRank - 2
        val outBatch = IntArray(batchRank) { i ->
            val aDim = if (i < aEffRank - 2) aEff[i] else 1
            val bDim = if (i < bEffRank - 2) bEff[i] else 1
            maxOf(aDim, bDim)
        }
        val m = aEff[aEffRank - 2]
        val n = bEff[bEffRank - 1]

        val outShape = when {
            aIs1D && bIs1D -> Shape(intArrayOf())
            aIs1D -> {
                val dims = IntArray(outBatch.size + 1)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 1] = n
                Shape(dims)
            }
            bIs1D -> {
                val dims = IntArray(outBatch.size + 1)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 1] = m
                Shape(dims)
            }
            else -> {
                val dims = IntArray(outBatch.size + 2)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 2] = m
                dims[dims.size - 1] = n
                Shape(dims)
            }
        }

        fun mapBatchIndexEff(batchIdx: IntArray, effDims: IntArray): IntArray {
            val inBatchRank = effDims.size - 2
            val mapped = IntArray(inBatchRank)
            var or = batchIdx.size - 1
            var ir = inBatchRank - 1
            while (ir >= 0) {
                val inDim = effDims[ir]
                val outIndex = if (or >= 0) batchIdx[or] else 0
                mapped[ir] = if (inDim == 1) 0 else outIndex
                ir--; or--
            }
            return mapped
        }

        // Safety net for TensorData implementations whose generic per-element get() doesn't
        // return the tensor's own dtype — e.g. a packed-quant weight wrapped in a
        // PreTransposedWeight marker (PreTransposedQ4_K/Q5_K/Q6_K/...), whose delegated get()
        // surfaces the raw packed byte rather than a dequantized Float. The dispatchers above
        // this fallback (chooseQuantizedMatmulHeap et al.) now route both prefill (rank > 2) and
        // single-token decode (rank == 1) activations to the packed-quant kernels, so this should
        // no longer be hit on that path — kept as defense in depth for any other TensorData
        // implementation with the same gap, materializing the dequantized array lazily (once,
        // only if actually needed) rather than up front for every matmulGeneric call.
        var aFallback: FloatArray? = null
        var bFallback: FloatArray? = null

        fun flatIndex(dims: IntArray, indices: IntArray): Int {
            var offset = 0
            for (i in dims.indices) offset = offset * dims[i] + indices[i]
            return offset
        }

        fun floatAt(data: TensorData<T, V>, dims: IntArray, indices: IntArray, isA: Boolean): Float {
            val raw = data.get(*indices)
            if (raw is Float) return raw
            val fallback = if (isA) {
                aFallback ?: data.copyToFloatArray().also { aFallback = it }
            } else {
                bFallback ?: data.copyToFloatArray().also { bFallback = it }
            }
            return fallback[flatIndex(dims, indices)]
        }

        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            val (batchIdx, mIdx, nIdx) = when {
                aIs1D && bIs1D -> Triple(IntArray(0), -1, -1)
                aIs1D -> {
                    val batchLen = outIdx.size - 1
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, -1, outIdx.last())
                }
                bIs1D -> {
                    val batchLen = outIdx.size - 1
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, outIdx.last(), -1)
                }
                else -> {
                    val batchLen = outIdx.size - 2
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, outIdx[outIdx.size - 2], outIdx[outIdx.size - 1])
                }
            }

            val aBatchIdx = mapBatchIndexEff(batchIdx, aEff)
            val bBatchIdx = mapBatchIndexEff(batchIdx, bEff)

            when (a.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    var k = 0
                    while (k < kA) {
                        val av: Float = if (aIs1D) {
                            floatAt(a.data, aDims, intArrayOf(k), isA = true)
                        } else {
                            val aIdx = IntArray(aRank)
                            if (aBatchIdx.isNotEmpty()) aBatchIdx.copyInto(aIdx)
                            aIdx[aRank - 2] = mIdx
                            aIdx[aRank - 1] = k
                            floatAt(a.data, aDims, aIdx, isA = true)
                        }
                        val bv: Float = if (bIs1D) {
                            floatAt(b.data, bDims, intArrayOf(k), isA = false)
                        } else {
                            val bIdx = IntArray(bRank)
                            if (bBatchIdx.isNotEmpty()) bBatchIdx.copyInto(bIdx)
                            bIdx[bRank - 2] = k
                            bIdx[bRank - 1] = nIdx
                            floatAt(b.data, bDims, bIdx, isA = false)
                        }
                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                sk.ainet.lang.types.Int32::class,
                sk.ainet.lang.types.Int8::class -> {
                    var acc = 0
                    var k = 0
                    while (k < kA) {
                        val av: Int = if (aIs1D) {
                            a.data.get(*intArrayOf(k)) as Int
                        } else {
                            val aIdx = IntArray(aRank)
                            if (aBatchIdx.isNotEmpty()) aBatchIdx.copyInto(aIdx)
                            aIdx[aRank - 2] = mIdx
                            aIdx[aRank - 1] = k
                            a.data.get(*aIdx) as Int
                        }
                        val bv: Int = if (bIs1D) {
                            b.data.get(*intArrayOf(k)) as Int
                        } else {
                            val bIdx = IntArray(bRank)
                            if (bBatchIdx.isNotEmpty()) bBatchIdx.copyInto(bIdx)
                            bIdx[bRank - 2] = k
                            bIdx[bRank - 1] = nIdx
                            b.data.get(*bIdx) as Int
                        }
                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for matmul: ${a.dtype}")
            }
        }
        return newTensor(outData, a.dtype, a, b)
    }

    /**
     * Weights already relayouted into kernel feed order, keyed by the identity of the packed bytes
     * they came from (#973/#1096).
     *
     * The relayout is O(bytes). Doing it inside [matmulWeightTransposed] once per weight instead of
     * once per call is what removes the per-forward copy `Linear.onForward` used to pay: a model's
     * weights are stable, so the first forward pass converts and every later one reuses. Bounded,
     * because a cache that grows without limit on a 2 GB device is its own bug; a model with more
     * than [PREPACK_CACHE_LIMIT] distinct packed weights simply converts the overflow each time,
     * which is exactly the old behaviour.
     */
    /**
     * `Wᵀ` for a **dense** 2-D float weight, materialized once per weight instead of once per call.
     *
     * `transpose` cannot hand back a view here: the kernels want the weight input-major
     * (`Fp32ViewMatmulKernel` declines anything whose `strides[0] != 1`), and no view of a
     * row-major `[out, in]` buffer has that layout — only a copy does. The copy itself is a
     * cache-hostile scatter over every element, so doing it per call is what actually costs: on
     * Gemma 4 E2B the two dense per-layer-embedding projections are transposed 70 times per token,
     * ~27M scattered element copies, and profiling attributed **84% of decode time** to those two
     * matmuls while the packed projections beside them — 8x larger — ran 22x faster.
     *
     * This is the dense counterpart of [prepackedWeights], which has cached the packed relayout
     * "once per weight instead of once per call" since #1096; the dense path simply never got the
     * same treatment. Keyed on buffer identity, so it holds only for the immutable parameter a
     * decode loop reuses; anything else falls through to a fresh transpose.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> transposedDenseWeight(weight: Tensor<T, V>): Tensor<T, V>? {
        if (weight.shape.rank != 2) return null
        // Key on the TensorData itself, not a FloatArray: a weight staged by the MemorySegment
        // factory is not FloatArray-backed, and that is precisely the case that hurts most —
        // `transpose` then falls all the way to its generic per-element fallback.
        val source: Any = weight.data
        transposedDenseWeights.firstOrNull { it.first === source }?.let { return it.second as Tensor<T, V> }
        // Materialize onto the HEAP, not through `transpose`/`dataFactory`. Two reasons: a
        // MemorySegment-backed dense weight sends `transpose` to its generic per-element fallback,
        // and — the bigger one — the vectorized FP32 kernels only accept heap `FloatArray`
        // operands, so a segment-backed transpose is condemned to the decoding reference kernel
        // afterwards. Since the cache pays for one copy anyway, it may as well land where the fast
        // kernels can read it.
        val transposed = heapTransposeFp32(weight) ?: transpose(weight)
        if (transposedDenseWeights.size >= TRANSPOSED_DENSE_CACHE_LIMIT) transposedDenseWeights.removeAt(0)
        transposedDenseWeights.add(Pair(source, transposed as Tensor<*, *>))
        return transposed
    }

    /** `Wᵀ` as a heap-backed dense FP32 tensor, or null when [weight] is not dense FP32. */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> heapTransposeFp32(weight: Tensor<T, V>): Tensor<T, V>? {
        if (weight.dtype != FP32::class) return null
        val rows = weight.shape[0]
        val cols = weight.shape[1]
        val src = runCatching { weight.data.copyToFloatArray() }.getOrNull() ?: return null
        if (src.size != rows * cols) return null
        val out = FloatArray(src.size)
        for (r in 0 until rows) {
            val base = r * cols
            for (c in 0 until cols) out[c * rows + r] = src[base + c]
        }
        return newTensor(
            sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(Shape(cols, rows), out)
                as sk.ainet.lang.tensor.data.TensorData<T, V>,
            weight.dtype,
            weight,
        )
    }

    private val transposedDenseWeights: MutableList<Pair<Any, Tensor<*, *>>> = mutableListOf()

    /** Big enough for a deep model's dense projections (Gemma 4 E2B has 70) without unbounded growth. */
    private val TRANSPOSED_DENSE_CACHE_LIMIT: Int = 256

    private val prepackedWeights: MutableList<Pair<ByteArray, Tensor<*, *>>> = mutableListOf()

    /** How many relayouted weights to keep; beyond this the oldest is dropped and reconverted on demand. */
    private val PREPACK_CACHE_LIMIT: Int = 64

    /**
     * `x · Wᵀ` with the weight as `[out, in]` — the primitive, and the way out of the per-forward
     * copy (#973 "the deeper semantic problem", #1096).
     *
     * For a block-quantized weight this relayouts **once** and reuses the result; for anything else
     * it is the ordinary `matmul(x, transpose(w))`, which for dense data is a free shape swap.
     */
    /**
     * The weight behind a `Wᵀ` marker (#1108), or `null` when [b] is an ordinary operand.
     *
     * `protected` because every `matmul` override has to consult it *before* its own fast paths:
     * the marker reports no packed storage by design, so any check that asks "is this packed?"
     * gets the wrong answer and a dense path would read the packed payload as floats.
     */
    protected fun <T : DType, V> untransposedWeight(b: Tensor<T, V>): Tensor<T, V>? {
        val marker = b.data as? TransposedWeightTensorData<T, V> ?: return null
        return newTensor(marker.weight, b.dtype, b)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        if (weight.shape.rank != 2 || !isHeapPackedWeight(weight.data)) {
            return matmul(x, transposedDenseWeight(weight) ?: transpose(weight))
        }
        // Decode the weight where it lies (#1124). The relayout below produces bytes for kernels
        // that address `packedData` in feed order themselves; this implementation has no such
        // kernel, so relayouting for it was pure harm — the result is a tensor whose shape says
        // [in, out] while its blocks still run along the original input dimension, which no block
        // order can describe, and decoding it read the wrong blocks and returned plausible garbage.
        // `KernelDispatch.matmul` already wants the weight output-major, which is exactly the shape
        // this weight has, so the canonical view goes straight in with nothing rearranged.
        matmulWeightTransposedViaViews(x, weight)?.let { return it }
        return matmul(x, transposePackedWeight(weight) ?: return matmulGeneric(x, transpose(weight)))
    }

    /**
     * `x · Wᵀ` with [weight] as `[out, in]`, computed by decoding through views (#1124).
     *
     * Correct for any packed encoding, because the reference kernel reads through the decoding
     * `get()`; slower than a packed kernel, which is why [DefaultCpuOpsJvm] overrides this with the
     * relayout-and-cache path its vectorized kernels can use. `null` when the operands cannot
     * describe themselves as views.
     */
    @Suppress("UNCHECKED_CAST")
    protected fun <T : DType, V> matmulWeightTransposedViaViews(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V>? {
        if (!sk.ainet.backend.api.kernel.DispatchMode.useRegistry()) return null
        if (x.dtype != FP32::class) return null
        val xView = x.data.view ?: return null
        val wView = weight.data.view ?: return null
        val (xNorm, leading) = try {
            sk.ainet.backend.api.kernel.KernelDispatch.normalizeActivation(xView)
        } catch (_: IllegalArgumentException) {
            return null
        }
        val m = xNorm.shape[0]
        val k = xNorm.shape[1]
        val n = wView.shape[0]                       // weight is [out, in]; the dispatcher wants it that way
        if (k != wView.shape[1]) return null
        val outArray = FloatArray(m * n)
        val outView = sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.Storage.Heap.wrap(outArray), Shape(m, n), FP32,
        )
        sk.ainet.backend.api.kernel.KernelDispatch.matmul(xNorm, wView, outView, dispatchScope())
        val outShape = when {
            x.shape.rank == 1 -> Shape(n)
            leading.isEmpty() -> Shape(m, n)
            else -> Shape(*(leading + n))
        }
        val outData = dataFactory.adoptFloatArray<T, Float>(outShape, x.dtype, outArray) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(outData, x.dtype, x, weight)
    }


    /**
     * The block-grid permutation that used to live in `transpose` (#973/#1096).
     *
     * The packed matmul kernels read `packedData` as **input-block-major** —
     * `(blockIdx * outputDim + o)`, every output row's block for one input block contiguous —
     * whatever shape the tensor declares. Canonical packed storage, as loaded from a GGUF or built
     * by any row-major producer, is the other order. The two coincide only at one block per row, so
     * for a real weight a bare shape relabel hands the kernel bytes in the wrong physical order and
     * it reads garbage without failing (#968, and downstream SKaiNET-transformers#307).
     *
     * So this is a real O(bytes) permutation, and [matmulWeightTransposed] runs it **once** per
     * weight rather than once per call — which is the difference #1096 exists to make.
     *
     * @return the relayouted weight, or `null` for a data type with no packed relayout
     */
    /** The heap packed data types whose kernels read input-block-major bytes. */
    private fun isHeapPackedWeight(data: sk.ainet.lang.tensor.data.TensorData<*, *>): Boolean =
        // Any packed block storage, not an enumeration of formats (#1136 physiology: kernels are
        // selected by the weight's storage format, and so is the Wᵀ-marker path that reaches them).
        // The hardcoded Q-list this replaces silently excluded the ternary types
        // (BitNetB158TensorData, BitNetPlanesTensorData), so `transpose` dense-copied them through
        // a decoding get() that returns CODES, not values — a ClassCastException at best. The JVM
        // tier keeps its own narrower list (isHeapPackedWeightForJvm) for the formats its
        // relayout-and-cache kernels actually read; everything else lands in
        // matmulWeightTransposedViaViews, which serves any encoding through KernelDispatch.
        data is sk.ainet.lang.tensor.storage.PackedBlockStorage

    /**
     * The block relayout by its own name (#973/#1096) — what `transpose` used to do to a packed
     * weight, for the callers that genuinely want the permuted bytes rather than a product.
     *
     * Prefer [matmulWeightTransposed], which does this once per weight instead of once per call.
     *
     * @throws UnsupportedOperationException for a data type with no packed relayout
     */
    override fun <T : DType, V> relayoutPackedWeightForKernels(weight: Tensor<T, V>): Tensor<T, V> =
        transposePackedWeight(weight)
            ?: throw UnsupportedOperationException(
                "no packed relayout for ${weight.data::class.simpleName}",
            )

    /**
     * A weight already stored in kernel feed order, relabelled as the `[in, out]` tensor the packed
     * kernels take — **sharing the same bytes** (#1120).
     *
     * This is the payoff of letting packed storage declare its order. `transposePackedWeight` is an
     * O(bytes) permutation run once per weight and cached; when the loader already produced feed
     * order there is nothing to permute, and all that is needed is the other shape label over the
     * same array. `null` when [tensor] is not a feed-order packed weight.
     */
    @Suppress("UNCHECKED_CAST")
    protected fun <T : DType, V> rewrapFeedOrderWeight(tensor: Tensor<T, V>): Tensor<T, V>? {
        if (tensor.shape.rank != 2) return null
        val packed = tensor.data as? sk.ainet.lang.tensor.storage.PackedBlockStorage ?: return null
        if (packed.blockOrder != sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR) return null
        val swapped = Shape(tensor.shape[1], tensor.shape[0])
        val bytes = packed.packedData
        val relabelled: TensorData<T, V> = when (tensor.data) {
            is Q4_KTensorData -> Q4_KBlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q5_KTensorData -> Q5_KBlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q6_KTensorData -> Q6_KBlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q5_1TensorData -> Q5_1BlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q5_0TensorData -> Q5_0BlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q8_0TensorData -> Q8_0BlockTensorData(swapped, bytes) as TensorData<T, V>
            is Q4_0TensorData -> Q4_0BlockTensorData(swapped, bytes) as TensorData<T, V>
            else -> return null
        }
        return newTensor(relabelled, tensor.dtype, tensor)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> transposePackedWeight(tensor: Tensor<T, V>): Tensor<T, V>? {
        val rank = tensor.shape.rank
        val rows = tensor.shape[rank - 2]
        val cols = tensor.shape[rank - 1]
            @Suppress("UNCHECKED_CAST")
            when (val d = tensor.data) {
                is Q4_KTensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q4_KTensorData.BLOCK_SIZE, "Q4_K")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q4_KTensorData.BYTES_PER_BLOCK)
                    return newTensor(Q4_KBlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                is Q5_KTensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q5_KTensorData.BLOCK_SIZE, "Q5_K")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q5_KTensorData.BYTES_PER_BLOCK)
                    return newTensor(Q5_KBlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                is Q6_KTensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q6_KTensorData.BLOCK_SIZE, "Q6_K")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q6_KTensorData.BYTES_PER_BLOCK)
                    return newTensor(Q6_KBlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                is Q5_1TensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q5_1TensorData.BLOCK_SIZE, "Q5_1")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q5_1TensorData.BYTES_PER_BLOCK)
                    return newTensor(Q5_1BlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                is Q5_0TensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q5_0TensorData.BLOCK_SIZE, "Q5_0")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q5_0TensorData.BYTES_PER_BLOCK)
                    return newTensor(Q5_0BlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                // Q8_0 / Q4_0: same physical block-grid transpose as the arms above —
                // this `when` covers every quant type chooseQuantizedMatmulHeap
                // dispatches, i.e. every packed type that can be a matmul weight
                // (originally retrofitted for the Byte→Float ClassCastException gap,
                // see transformers #178; the shape-swap-only version of these arms
                // carried the same block-order bug as Q5_0/Q5_1 above).
                is Q8_0TensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q8_0TensorData.BLOCK_SIZE, "Q8_0")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q8_0TensorData.BYTES_PER_BLOCK)
                    return newTensor(Q8_0BlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                is Q4_0TensorData -> {
                    val blocksPerInputDim = requirePackedBlockAligned(cols, Q4_0TensorData.BLOCK_SIZE, "Q4_0")
                    val reordered = transposePackedBlocks(d.packedData, rows, blocksPerInputDim, Q4_0TensorData.BYTES_PER_BLOCK)
                    return newTensor(Q4_0BlockTensorData(Shape(cols, rows), reordered) as TensorData<T, V>, tensor.dtype, tensor)
                }
                else -> {}
            }

        return null
    }

    @TensorOp()
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val rank = tensor.shape.rank
        require(rank >= 2) { "Transpose requires at least 2 dimensions" }
        val rows = tensor.shape[rank - 2]
        val cols = tensor.shape[rank - 1]

        // Transposing a transposed weight is the weight (#1108). Restoring this involution is half
        // the point of the marker: `w.t().t()` has not been `w` for as long as packed data existed.
        val alreadyTransposed = tensor.data as? TransposedWeightTensorData<T, V>
        if (alreadyTransposed != null) {
            return newTensor(alreadyTransposed.weight, tensor.dtype, tensor)
        }

        // Only the *heap* packed types, whose kernels read input-block-major bytes. The
        // MemorySegment tier reads canonical bytes, so for those a shape swap is genuinely correct
        // and stays where it is — census contradiction #3, now stated instead of implied.
        val heapPacked = rank == 2 && isHeapPackedWeight(tensor.data)
        if (heapPacked) {
            // Transposing block-quantized bytes is still not a representable operation (#973):
            // blocks quantize runs along the input dimension, so a real transpose needs
            // requantization. #1096 answered that by making `x · Wᵀ` a primitive and having this
            // throw; #1108 keeps the primitive and drops the throw, because refusing here forced
            // the *caller* to know how the weight was stored — which depends on the file loaded and
            // the device it runs on, not on the model.
            //
            // So this returns `Wᵀ` as an unmaterialized marker instead. It carries no `packedData`,
            // so no kernel can read the bytes through it as if they were the transpose's; `matmul`
            // recognises it and asks `matmulWeightTransposed` for the product, which relayouts
            // properly and once per weight. That is the distinction from the relabel this replaced.
            return newTensor(TransposedWeightTensorData(tensor.data), tensor.dtype, tensor)
        }

        // Narrow floats (FP16/BF16) relaid input-major at load are a *view* rewrap, not a packed
        // relayout: the transpose is the same buffer read with the other shape's strides, which is
        // what lets a KEEP_NATIVE weight survive Linear's weight handling and reach the narrow
        // matmul kernel (#888). Only the block-quantized types are refused above (#973).
        if (rank == 2) {
            val narrow = tensor.data as? NarrowFloatInputMajorTensorData
            if (narrow != null) {
                @Suppress("UNCHECKED_CAST")
                return newTensor(narrow.transposedView() as TensorData<T, V>, tensor.dtype, tensor)
            }
        }

        // Fast path: 2D float tensor — direct buffer swap
        if (rank == 2 && tensor.data is FloatArrayTensorData<*>) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val out = FloatArray(rows * cols)
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    out[c * rows + r] = buf[r * cols + c]
                }
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.adoptFloatArray<T, Float>(Shape(cols, rows), tensor.dtype, out) as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }

        // Generic fallback
        val outDims = tensor.shape.dimensions.copyOf()
        outDims[rank - 1] = rows
        outDims[rank - 2] = cols
        val outShape = Shape(outDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
            val inIdx = outIdx.copyOf()
            inIdx[rank - 2] = outIdx[rank - 1]
            inIdx[rank - 1] = outIdx[rank - 2]
            tensor.data.get(*inIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    override fun <T : DType, V> permute(tensor: Tensor<T, V>, axes: IntArray): Tensor<T, V> {
        val rank = tensor.shape.rank
        require(axes.size == rank) {
            "permute: axes length ${axes.size} must match tensor rank $rank"
        }
        val seen = BooleanArray(rank)
        for (a in axes) {
            require(a in 0 until rank) { "permute: axis $a out of range [0, $rank)" }
            require(!seen[a]) { "permute: axis $a appears more than once in ${axes.toList()}" }
            seen[a] = true
        }

        val inDims = tensor.shape.dimensions
        val outDims = IntArray(rank) { i -> inDims[axes[i]] }
        val outShape = Shape(outDims)

        // Identity permute — no copy.
        var isIdentity = true
        for (i in 0 until rank) if (axes[i] != i) { isIdentity = false; break }
        if (isIdentity) return tensor

        // Row-major strides for input and output. inStrides[k] is the
        // distance in the source buffer between consecutive indices on
        // input axis k.
        val inStrides = IntArray(rank).also { s ->
            s[rank - 1] = 1
            for (i in rank - 2 downTo 0) s[i] = s[i + 1] * inDims[i + 1]
        }
        val outStrides = IntArray(rank).also { s ->
            s[rank - 1] = 1
            for (i in rank - 2 downTo 0) s[i] = s[i + 1] * outDims[i + 1]
        }

        // Fast path: source is a contiguous FloatArray. Iterate the output
        // linearly, decompose each flat index to its multi-index, permute
        // to source coords, recompose to source flat index, copy.
        if (tensor.data is FloatArrayTensorData<*>) {
            val srcBuf = (tensor.data as FloatArrayTensorData<*>).buffer
            val total = outShape.volume
            val out = FloatArray(total)
            val outIdx = IntArray(rank)
            for (flatOut in 0 until total) {
                var rem = flatOut
                for (i in 0 until rank) {
                    val s = outStrides[i]
                    outIdx[i] = rem / s
                    rem -= outIdx[i] * s
                }
                var flatIn = 0
                for (i in 0 until rank) flatIn += outIdx[i] * inStrides[axes[i]]
                out[flatOut] = srcBuf[flatIn]
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.adoptFloatArray<T, Float>(outShape, tensor.dtype, out)
                as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }

        // Generic fallback: defer to dataFactory.init with element access.
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
            val inIdx = IntArray(rank)
            for (i in 0 until rank) inIdx[axes[i]] = outIdx[i]
            tensor.data.get(*inIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-conv2d")
    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 4) { "conv2d: input must be 4D (N, C_in, H, W), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 4) { "conv2d: weight must be 4D (C_out, C_in/groups, kH, kW), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv2d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv2d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv2d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kH = weight.shape[2]
        val kW = weight.shape[3]

        require(cIn % groups == 0) { "conv2d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv2d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv2d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        val (sH, sW) = stride
        val (pH, pW) = padding
        val (dH, dW) = dilation

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outH = outDim(inH, kH, sH, pH, dH)
        val outW = outDim(inW, kW, sW, pW, dW)
        require(outH >= 0 && outW >= 0) { "conv2d: computed negative output shape (H=${outH}, W=${outW})" }

        // Validate bias shape if provided (accept [C_out] or [1,C_out,1,1])
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv2d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                4 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1 && bias.shape[3] == 1) {
                        "conv2d: bias shape must be [1,C_out,1,1] when 4D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv2d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kh = 0
                        while (kh < kH) {
                            val ih = hBase + kh * dH
                            if (ih >= 0 && ih < inH) {
                                var kw = 0
                                while (kw < kW) {
                                    val iw = wBase + kw * dW
                                    if (iw >= 0 && iw < inW) {
                                        val vIn = input.data.get(bIdx, ic, ih, iw) as Float
                                        val vW = weight.data.get(oc, kc, kh, kw) as Float
                                        acc += vIn * vW
                                    }
                                    kw++
                                }
                            }
                            kh++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            4 -> bias.data.get(0, oc, 1 - 1, 1 - 1) as Float // [1, C_out, 1, 1]
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kh = 0
                        while (kh < kH) {
                            val ih = hBase + kh * dH
                            if (ih >= 0 && ih < inH) {
                                var kw = 0
                                while (kw < kW) {
                                    val iw = wBase + kw * dW
                                    if (iw >= 0 && iw < inW) {
                                        val vIn = input.data.get(bIdx, ic, ih, iw) as Int
                                        val vW = weight.data.get(oc, kc, kh, kw) as Int
                                        acc += vIn * vW
                                    }
                                    kw++
                                }
                            }
                            kh++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            4 -> bias.data.get(0, oc, 0, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv2d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    @TensorOp()
    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 3) { "conv1d: input must be 3D (N, C_in, L), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 3) { "conv1d: weight must be 3D (C_out, C_in/groups, K), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv1d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv1d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv1d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inL = input.shape[2]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kL = weight.shape[2]

        require(cIn % groups == 0) { "conv1d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv1d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv1d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outL = outDim(inL, kL, stride, padding, dilation)
        require(outL >= 0) { "conv1d: computed negative output length (L=${outL})" }

        // Validate bias shape if provided
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv1d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                3 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1) {
                        "conv1d: bias shape must be [1,C_out,1] when 3D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv1d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outL)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val ol = outIdx[2]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val lBase = ol * stride - padding

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kl = 0
                        while (kl < kL) {
                            val il = lBase + kl * dilation
                            if (il >= 0 && il < inL) {
                                val vIn = input.data.get(bIdx, ic, il) as Float
                                val vW = weight.data.get(oc, kc, kl) as Float
                                acc += vIn * vW
                            }
                            kl++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            3 -> bias.data.get(0, oc, 0) as Float
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val lBase = ol * stride - padding

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kl = 0
                        while (kl < kL) {
                            val il = lBase + kl * dilation
                            if (il >= 0 && il < inL) {
                                val vIn = input.data.get(bIdx, ic, il) as Int
                                val vW = weight.data.get(oc, kc, kl) as Int
                                acc += vIn * vW
                            }
                            kl++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            3 -> bias.data.get(0, oc, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv1d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    override fun <T : DType, V> convTranspose1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        outputPadding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        // input: [batch, inChannels, inLength]
        // weight: [inChannels, outChannels/groups, kernelSize]
        val batch = input.shape[0]
        val inChannels = input.shape[1]
        val inLength = input.shape[2]
        val outChannelsPerGroup = weight.shape[1]
        val kernelSize = weight.shape[2]
        val outChannels = outChannelsPerGroup * groups
        val outLength = (inLength - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1

        val outData = dataFactory.zeros<T, V>(Shape(batch, outChannels, outLength), input.dtype)

        val inData = input.data
        val wData = weight.data

        val inChPerGroup = inChannels / groups

        for (b in 0 until batch) {
            for (g in 0 until groups) {
                for (ic in 0 until inChPerGroup) {
                    for (oc in 0 until outChannelsPerGroup) {
                        for (il in 0 until inLength) {
                            val inputVal = inData.get(b, g * inChPerGroup + ic, il) as Float
                            if (inputVal == 0f) continue
                            for (k in 0 until kernelSize) {
                                val ol = il * stride - padding + k * dilation
                                if (ol < 0 || ol >= outLength) continue
                                val weightVal = wData.get(g * inChPerGroup + ic, oc, k) as Float
                                val existing = outData.get(b, g * outChannelsPerGroup + oc, ol) as Float
                                @Suppress("UNCHECKED_CAST")
                                outData.set(b, g * outChannelsPerGroup + oc, ol, value = (existing + inputVal * weightVal) as V)
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if (bias != null) {
            val biasData = bias.data
            for (b in 0 until batch) {
                for (oc in 0 until outChannels) {
                    val biasVal = biasData.get(oc) as Float
                    for (ol in 0 until outLength) {
                        val existing = outData.get(b, oc, ol) as Float
                        @Suppress("UNCHECKED_CAST")
                        outData.set(b, oc, ol, value = (existing + biasVal) as V)
                    }
                }
            }
        }

        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 5) { "conv3d: input must be 5D (N, C_in, D, H, W), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 5) { "conv3d: weight must be 5D (C_out, C_in/groups, kD, kH, kW), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv3d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv3d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv3d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inD = input.shape[2]
        val inH = input.shape[3]
        val inW = input.shape[4]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kD = weight.shape[2]
        val kH = weight.shape[3]
        val kW = weight.shape[4]

        require(cIn % groups == 0) { "conv3d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv3d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv3d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        val (sD, sH, sW) = stride
        val (pD, pH, pW) = padding
        val (dD, dH, dW) = dilation

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outD = outDim(inD, kD, sD, pD, dD)
        val outH = outDim(inH, kH, sH, pH, dH)
        val outW = outDim(inW, kW, sW, pW, dW)
        require(outD >= 0 && outH >= 0 && outW >= 0) { "conv3d: computed negative output shape (D=${outD}, H=${outH}, W=${outW})" }

        // Validate bias shape if provided
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv3d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                5 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1 && bias.shape[3] == 1 && bias.shape[4] == 1) {
                        "conv3d: bias shape must be [1,C_out,1,1,1] when 5D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv3d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outD, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val od = outIdx[2]
            val oh = outIdx[3]
            val ow = outIdx[4]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val dBase = od * sD - pD
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kd = 0
                        while (kd < kD) {
                            val id = dBase + kd * dD
                            if (id >= 0 && id < inD) {
                                var kh = 0
                                while (kh < kH) {
                                    val ih = hBase + kh * dH
                                    if (ih >= 0 && ih < inH) {
                                        var kw = 0
                                        while (kw < kW) {
                                            val iw = wBase + kw * dW
                                            if (iw >= 0 && iw < inW) {
                                                val vIn = input.data.get(bIdx, ic, id, ih, iw) as Float
                                                val vW = weight.data.get(oc, kc, kd, kh, kw) as Float
                                                acc += vIn * vW
                                            }
                                            kw++
                                        }
                                    }
                                    kh++
                                }
                            }
                            kd++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            5 -> bias.data.get(0, oc, 0, 0, 0) as Float
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val dBase = od * sD - pD
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kd = 0
                        while (kd < kD) {
                            val id = dBase + kd * dD
                            if (id >= 0 && id < inD) {
                                var kh = 0
                                while (kh < kH) {
                                    val ih = hBase + kh * dH
                                    if (ih >= 0 && ih < inH) {
                                        var kw = 0
                                        while (kw < kW) {
                                            val iw = wBase + kw * dW
                                            if (iw >= 0 && iw < inW) {
                                                val vIn = input.data.get(bIdx, ic, id, ih, iw) as Int
                                                val vW = weight.data.get(oc, kc, kd, kh, kw) as Int
                                                acc += vIn * vW
                                            }
                                            kw++
                                        }
                                    }
                                    kh++
                                }
                            }
                            kd++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            5 -> bias.data.get(0, oc, 0, 0, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv3d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-upsample2d")
    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> {
        require(input.rank == 4) { "upsample2d: input must be 4D (N, C, H, W)" }
        val (scaleH, scaleW) = scale
        require(scaleH > 0 && scaleW > 0) { "upsample2d: scale factors must be positive" }

        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val outH = inH * scaleH
        val outW = inW * scaleW
        val outShape = Shape(n, c, outH, outW)

        val outData = when (mode) {
            UpsampleMode.Nearest -> dataFactory.init<T, V>(outShape, input.dtype) { idx ->
                val oh = idx[2]
                val ow = idx[3]
                val ih = oh / scaleH
                val iw = ow / scaleW
                input.data.get(idx[0], idx[1], ih, iw)
            }

            UpsampleMode.Bilinear -> {
                require(input.dtype == FP32::class || input.dtype == FP16::class) {
                    "upsample2d: Bilinear mode is only implemented for float dtypes (got ${input.dtype})"
                }
                dataFactory.init<T, V>(outShape, input.dtype) { idx ->
                    val b = idx[0]
                    val ch = idx[1]
                    val srcH = sourceCoord(idx[2], scaleH, inH, alignCorners)
                    val srcW = sourceCoord(idx[3], scaleW, inW, alignCorners)
                    val ih0 = floor(srcH).toInt().coerceIn(0, inH - 1)
                    val ih1 = (ih0 + 1).coerceIn(0, inH - 1)
                    val iw0 = floor(srcW).toInt().coerceIn(0, inW - 1)
                    val iw1 = (iw0 + 1).coerceIn(0, inW - 1)
                    val wh = (srcH - ih0).coerceIn(0.0f, 1.0f)
                    val ww = (srcW - iw0).coerceIn(0.0f, 1.0f)
                    val v00 = (input.data.get(b, ch, ih0, iw0) as Number).toFloat()
                    val v01 = (input.data.get(b, ch, ih0, iw1) as Number).toFloat()
                    val v10 = (input.data.get(b, ch, ih1, iw0) as Number).toFloat()
                    val v11 = (input.data.get(b, ch, ih1, iw1) as Number).toFloat()
                    val blend = v00 * (1f - wh) * (1f - ww) +
                        v01 * (1f - wh) * ww +
                        v10 * wh * (1f - ww) +
                        v11 * wh * ww
                    @Suppress("UNCHECKED_CAST")
                    (blend as V)
                }
            }
        }
        return newTensor(outData, input.dtype, input)
    }

    /**
     * Maps an output coordinate to the (fractional) source coordinate for upsampling,
     * matching the PyTorch convention. With [alignCorners] = false the sample centers are
     * `(o + 0.5) / scale - 0.5`; with align corners the endpoints are pinned via
     * `o * (in - 1) / (out - 1)`. The result may fall outside `[0, in-1]`; callers clamp.
     */
    private fun sourceCoord(out: Int, scale: Int, inDim: Int, alignCorners: Boolean): Float {
        val outDim = inDim * scale
        return if (alignCorners) {
            if (outDim <= 1) 0f else out.toFloat() * (inDim - 1) / (outDim - 1)
        } else {
            (out + 0.5f) / scale - 0.5f
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-maxpool2d")
    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        require(input.rank == 4) { "maxPool2d: input must be 4D (N, C, H, W)" }
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (pH, pW) = padding
        require(kH > 0 && kW > 0) { "maxPool2d: kernel must be > 0" }
        require(sH > 0 && sW > 0) { "maxPool2d: stride must be > 0" }
        fun outDim(inDim: Int, k: Int, s: Int, p: Int): Int = ((inDim + 2 * p - k) / s) + 1
        val outH = outDim(inH, kH, sH, pH)
        val outW = outDim(inW, kW, sW, pW)
        require(outH >= 0 && outW >= 0) { "maxPool2d: negative output size (H=${outH}, W=${outW})" }
        val outShape = Shape(n, c, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val ch = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]
            val hBase = oh * sH - pH
            val wBase = ow * sW - pW
            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var best = Float.NEGATIVE_INFINITY
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        if (ih in 0 until inH) {
                            var kw = 0
                            while (kw < kW) {
                                val iw = wBase + kw
                                if (iw in 0 until inW) {
                                    val v = input.data.get(bIdx, ch, ih, iw) as Float
                                    if (v > best) best = v
                                }
                                kw++
                            }
                        }
                        kh++
                    }
                    @Suppress("UNCHECKED_CAST")
                    best as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    var best = Int.MIN_VALUE
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        if (ih in 0 until inH) {
                            var kw = 0
                            while (kw < kW) {
                                val iw = wBase + kw
                                if (iw in 0 until inW) {
                                    val v = input.data.get(bIdx, ch, ih, iw) as Int
                                    if (v > best) best = v
                                }
                                kw++
                            }
                        }
                        kh++
                    }
                    @Suppress("UNCHECKED_CAST")
                    best as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for maxPool2d: ${input.dtype}")
            }
        }
        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    override fun <T : DType, V> avgPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean
    ): Tensor<T, V> {
        require(input.rank == 4) { "avgPool2d: input must be 4D (N, C, H, W)" }
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (pH, pW) = padding
        require(kH > 0 && kW > 0) { "avgPool2d: kernel must be > 0" }
        require(sH > 0 && sW > 0) { "avgPool2d: stride must be > 0" }
        fun outDim(inDim: Int, k: Int, s: Int, p: Int): Int = ((inDim + 2 * p - k) / s) + 1
        val outH = outDim(inH, kH, sH, pH)
        val outW = outDim(inW, kW, sW, pW)
        require(outH >= 0 && outW >= 0) { "avgPool2d: negative output size (H=${outH}, W=${outW})" }
        val outShape = Shape(n, c, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val ch = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]
            val hBase = oh * sH - pH
            val wBase = ow * sW - pW
            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var sum = 0.0f
                    var count = 0
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        var kw = 0
                        while (kw < kW) {
                            val iw = wBase + kw
                            if (ih in 0 until inH && iw in 0 until inW) {
                                val v = input.data.get(bIdx, ch, ih, iw) as Float
                                sum += v
                                count++
                            } else if (countIncludePad) {
                                // Include padding as zero in the count
                                count++
                            }
                            kw++
                        }
                        kh++
                    }
                    // If countIncludePad is true but no valid elements, use kernel size
                    val divisor = if (countIncludePad) (kH * kW) else maxOf(count, 1)
                    @Suppress("UNCHECKED_CAST")
                    (sum / divisor) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    var sum = 0
                    var count = 0
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        var kw = 0
                        while (kw < kW) {
                            val iw = wBase + kw
                            if (ih in 0 until inH && iw in 0 until inW) {
                                val v = input.data.get(bIdx, ch, ih, iw) as Int
                                sum += v
                                count++
                            } else if (countIncludePad) {
                                count++
                            }
                            kw++
                        }
                        kh++
                    }
                    val divisor = if (countIncludePad) (kH * kW) else maxOf(count, 1)
                    @Suppress("UNCHECKED_CAST")
                    (sum / divisor) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for avgPool2d: ${input.dtype}")
            }
        }
        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-reshape")
    override fun <T : DType, V> reshape(
        tensor: Tensor<T, V>,
        newShape: Shape
    ): Tensor<T, V> {
        // Support -1 dimension inference and validate total elements
        val dims = newShape.dimensions.copyOf()
        var negOneIdx = -1
        var knownProduct = 1
        for (i in dims.indices) {
            val d = dims[i]
            if (d == -1) {
                require(negOneIdx == -1) { "Only one dimension can be -1 in reshape" }
                negOneIdx = i
            } else {
                require(d >= 0) { "Shape dims must be >=0 or -1 for inference: ${d}" }
                knownProduct *= if (d == 0 && dims.size == 0) 1 else d
            }
        }
        val inVol = tensor.shape.volume
        if (negOneIdx >= 0) {
            require(knownProduct != 0) { "Cannot infer dimension with zero known product" }
            require(inVol % knownProduct == 0) { "Cannot infer dimension: input volume ${inVol} not divisible by known product ${knownProduct}" }
            dims[negOneIdx] = inVol / knownProduct
        }
        val finalShape = Shape(dims)
        require(finalShape.volume == inVol) { "Reshape volume mismatch: input=${inVol}, output=${finalShape.volume}" }
        floatBufferOf(tensor)?.let { src ->
            return floatResult(finalShape, tensor.dtype, src.copyOf(), tensor)
        }
        // Reinitialize data by mapping flat index order
        val outData = dataFactory.init<T, V>(finalShape, tensor.dtype) { outIdx ->
            // Compute flat index in output (row-major)
            val outStrides = IntArray(finalShape.rank).apply {
                var s = 1
                for (i in finalShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= finalShape[i]
                }
            }
            var flat = 0
            for (i in outIdx.indices) flat += outIdx[i] * outStrides[i]
            // Map flat index to input indices using input shape strides
            val inShape = tensor.shape
            val inStrides = IntArray(inShape.rank).apply {
                var s = 1
                for (i in inShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= inShape[i]
                }
            }
            val inIdx = IntArray(inShape.rank)
            var rem = flat
            for (i in 0 until inShape.rank) {
                inIdx[i] = rem / inStrides[i]
                rem %= inStrides[i]
            }
            tensor.data.get(*inIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-flatten")
    override fun <T : DType, V> flatten(
        tensor: Tensor<T, V>,
        startDim: Int,
        endDim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank >= 0) { "Invalid tensor rank" }
        fun normDim(d: Int, allowEqRank: Boolean = false): Int {
            val max = if (allowEqRank) rank else rank - 1
            val nd = if (d < 0) d + rank else d
            require(nd in 0..max) { "Dimension out of range: ${d} for rank ${rank}" }
            return nd
        }
        val s = normDim(startDim)
        val e = if (endDim == -1) rank - 1 else normDim(endDim)
        require(s <= e) { "startDim must be <= endDim: start=${s}, end=${e}" }
        if (rank == 0) return tensor // scalar no-op
        // Build new shape
        val newDims = mutableListOf<Int>()
        for (i in 0 until s) newDims += tensor.shape[i]
        var prod = 1
        for (i in s..e) prod *= tensor.shape[i]
        newDims += prod
        for (i in e + 1 until rank) newDims += tensor.shape[i]
        return reshape(tensor, Shape(newDims.toIntArray()))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-concat")
    override fun <T : DType, V> concat(
        tensors: List<Tensor<T, V>>,
        dim: Int
    ): Tensor<T, V> {
        require(tensors.isNotEmpty()) { "concat: tensors list must not be empty" }
        val first = tensors.first()
        val rank = first.rank
        // Normalize dim allowing dim==rank for scalars to create 1D
        val nd = if (dim < 0) dim + maxOf(rank, 1) else dim
        require(nd >= 0 && nd <= rank) { "concat: dim ${dim} out of range for rank ${rank}" }
        // Allow concatenation along any valid dimension (including dim 0 for rank > 1)
        // Validate shapes and dtype, compute output dims
        var concatSize = 0
        val outDims = IntArray(if (rank == 0) 1 else rank) { i -> if (rank == 0) 0 else first.shape[i] }
        tensors.forEachIndexed { idx, t ->
            require(t.dtype == first.dtype) { "concat: dtype mismatch at tensor ${idx}" }
            if (rank == 0) {
                // scalars: treat as 1D concat
                concatSize += 1
            } else {
                require(t.rank == rank) { "concat: rank mismatch at tensor ${idx}" }
                for (i in 0 until rank) {
                    if (i == nd) continue
                    require(t.shape[i] == first.shape[i]) { "concat: shape mismatch at dim ${i} for tensor ${idx}" }
                }
                concatSize += t.shape[nd]
            }
        }
        if (rank == 0) {
            outDims[0] = concatSize
        } else {
            outDims[nd] = concatSize
        }
        val outShape = Shape(outDims)
        val dtype = first.dtype
        val prefixSums = IntArray(tensors.size + 1)
        for (i in tensors.indices) {
            val sz = if (rank == 0) 1 else tensors[i].shape[nd]
            prefixSums[i + 1] = prefixSums[i] + sz
        }
        if (rank > 0 && nd < rank) {
            var allFloat = true
            val buffers = arrayOfNulls<FloatArray>(tensors.size)
            for (ti in tensors.indices) {
                val fb = floatBufferOf(tensors[ti])
                if (fb == null) { allFloat = false; break }
                buffers[ti] = fb
            }
            if (allFloat) {
                var inner = 1
                for (i2 in nd + 1 until rank) inner *= first.shape[i2]
                var outer = 1
                for (i2 in 0 until nd) outer *= first.shape[i2]
                val out = FloatArray(outShape.volume)
                var dst = 0
                for (o in 0 until outer) {
                    for (ti in tensors.indices) {
                        val block = tensors[ti].shape[nd] * inner
                        val srcOff = o * block
                        buffers[ti]!!.copyInto(out, dst, srcOff, srcOff + block)
                        dst += block
                    }
                }
                return floatResult(outShape, dtype, out, *tensors.toTypedArray())
            }
        }
        val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
            var k = 0
            val along = if (rank == 0) outIdx[0] else outIdx[nd]
            while (k < tensors.size && prefixSums[k + 1] <= along) k++
            val src = tensors[k]
            val localIdx = along - prefixSums[k]
            val inIdx = if (rank == 0) IntArray(0) else outIdx.copyOf()
            if (rank != 0) inIdx[nd] = localIdx
            src.data.get(*inIdx)
        }
        return newTensor(outData, dtype, *tensors.toTypedArray())
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-split")
    override fun <T : DType, V> split(
        tensor: Tensor<T, V>,
        splitSize: Int,
        dim: Int
    ): List<Tensor<T, V>> {
        require(splitSize > 0) { "split: splitSize must be > 0" }
        val rank = tensor.rank
        require(rank >= 0) { "split: invalid rank" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "split: dim ${dim} out of range for rank ${rank}" }
        val total = tensor.shape[nd]
        val chunks = (total + splitSize - 1) / splitSize
        val result = ArrayList<Tensor<T, V>>(chunks)
        var offset = 0
        for (c in 0 until chunks) {
            val size = minOf(splitSize, total - offset)
            val newDims = tensor.shape.dimensions.copyOf()
            newDims[nd] = size
            val outShape = Shape(newDims)
            val dtype = tensor.dtype
            val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
                val inIdx = outIdx.copyOf()
                inIdx[nd] = inIdx[nd] + offset
                tensor.data.get(*inIdx)
            }
            result += newTensor(outData, dtype, tensor)
            offset += size
        }
        return result
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-squeeze")
    override fun <T : DType, V> squeeze(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank > 0) { "squeeze: tensor must have rank > 0" }
        val dims = tensor.shape.dimensions
        val newDims = if (dim == null) {
            val kept = dims.filter { it != 1 }
            if (kept.isEmpty()) intArrayOf(1) else kept.toIntArray()
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "squeeze: dim ${dim} out of range for rank ${rank}" }
            require(dims[nd] == 1) { "squeeze: dimension ${dim} must be of size 1" }
            val list = dims.toMutableList()
            list.removeAt(nd)
            if (list.isEmpty()) intArrayOf(1) else list.toIntArray()
        }
        if (newDims.contentEquals(dims)) return tensor
        return reshape(tensor, Shape(newDims))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-unsqueeze")
    override fun <T : DType, V> unsqueeze(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        val nd = if (dim < 0) dim + (rank + 1) else dim
        require(nd in 0..rank) { "unsqueeze: dim ${dim} out of range for rank ${rank}" }
        val newDims = IntArray(rank + 1)
        for (i in 0 until nd) newDims[i] = tensor.shape[i]
        newDims[nd] = 1
        for (i in nd until rank) newDims[i + 1] = tensor.shape[i]
        return reshape(tensor, Shape(newDims))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-relu")
    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x < 0f) 0f else x }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0f) 0f else v) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0) 0 else v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for relu: ${'$'}{tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    override fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x < 0f) negativeSlope * x else x }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0f) negativeSlope * v else v) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0) (negativeSlope * v).toInt() else v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for leakyRelu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    override fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x >= 0f) x else alpha * (kotlin.math.exp(x) - 1f) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= 0f) v else alpha * (kotlin.math.exp(v) - 1f)) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for elu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-softmax")
    override fun <T : DType, V> softmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        // Stable softmax along dim
        val rank = tensor.rank
        require(rank > 0) { "softmax: tensor must have rank > 0" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "softmax: dim ${dim} out of range for rank ${rank}" }
        if (nd == rank - 1) {
            val src = floatBufferOf(tensor)
            val last = tensor.shape[nd]
            if (src != null && last > 0) {
                val rows = src.size / last
                val out = FloatArray(src.size)
                for (r in 0 until rows) {
                    val off = r * last
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until last) if (src[off + k] > maxVal) maxVal = src[off + k]
                    var denom = 0.0f
                    for (k in 0 until last) {
                        val e = kotlin.math.exp(src[off + k] - maxVal)
                        out[off + k] = e
                        denom += e
                    }
                    for (k in 0 until last) out[off + k] /= denom
                }
                return floatResult(tensor.shape, tensor.dtype, out, tensor)
            }
        }
        when (tensor.dtype) {
            sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
                    // For the slice defined by outIdx except for nd, compute softmax at that position
                    val idxMax = outIdx.copyOf()
                    // compute max along nd
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until tensor.shape[nd]) {
                        idxMax[nd] = k
                        val x = tensor.data.get(*idxMax) as Float
                        if (x > maxVal) maxVal = x
                    }
                    // compute denom
                    var denom = 0.0f
                    val idxDen = outIdx.copyOf()
                    for (k in 0 until tensor.shape[nd]) {
                        idxDen[nd] = k
                        val x = tensor.data.get(*idxDen) as Float
                        denom += kotlin.math.exp(x - maxVal)
                    }
                    // numerator for current position
                    val xOut = tensor.data.get(*outIdx) as Float
                    val num = kotlin.math.exp(xOut - maxVal)
                    @Suppress("UNCHECKED_CAST")
                    (num / denom) as V
                }
                return newTensor(outData, tensor.dtype, tensor)
            }
            else -> throw IllegalArgumentException("Unsupported dtype for softmax: ${tensor.dtype}")
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-logsoftmax")
    override fun <T : DType, V> logSoftmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank > 0) { "logSoftmax: tensor must have rank > 0" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "logSoftmax: dim ${dim} out of range for rank ${rank}" }
        if (nd == rank - 1) {
            val src = floatBufferOf(tensor)
            val last = tensor.shape[nd]
            if (src != null && last > 0) {
                val rows = src.size / last
                val out = FloatArray(src.size)
                for (r in 0 until rows) {
                    val off = r * last
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until last) if (src[off + k] > maxVal) maxVal = src[off + k]
                    var sumExp = 0.0f
                    for (k in 0 until last) {
                        sumExp += kotlin.math.exp((src[off + k] - maxVal).toDouble()).toFloat()
                    }
                    val logSumExp = kotlin.math.ln(sumExp.toDouble()).toFloat() + maxVal
                    for (k in 0 until last) out[off + k] = src[off + k] - logSumExp
                }
                return floatResult(tensor.shape, tensor.dtype, out, tensor)
            }
        }
        when (tensor.dtype) {
            sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
                    val idx = outIdx.copyOf()
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until tensor.shape[nd]) {
                        idx[nd] = k
                        val x = tensor.data.get(*idx) as Float
                        if (x > maxVal) maxVal = x
                    }
                    var sumExp = 0.0f
                    for (k in 0 until tensor.shape[nd]) {
                        idx[nd] = k
                        val x = tensor.data.get(*idx) as Float
                        sumExp += kotlin.math.exp((x - maxVal).toDouble()).toFloat()
                    }
                    val logSumExp = kotlin.math.ln(sumExp.toDouble()).toFloat() + maxVal
                    val xOut = tensor.data.get(*outIdx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (xOut - logSumExp) as V
                }
                return newTensor(outData, tensor.dtype, tensor)
            }
            else -> throw IllegalArgumentException("Unsupported dtype for logSoftmax: ${tensor.dtype}")
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sigmoid")
    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> 1.0f / (1.0f + kotlin.math.exp(-x)) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (1.0f / (1.0f + kotlin.math.exp(-x))) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for sigmoid: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-silu")
    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> x * (1.0f / (1.0f + kotlin.math.exp(-x))) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    val s = 1.0f / (1.0f + kotlin.math.exp(-x))
                    @Suppress("UNCHECKED_CAST")
                    (x * s) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for silu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-gelu")
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x ->
            val inner = x + 0.044715f * (x * x * x)
            0.5f * x * (1.0f + kotlin.math.tanh(0.7978845608f * inner))
        }?.let { return it }
        // Tanh approximation of GELU
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    val x3 = x * x * x
                    val inner = x + 0.044715f * x3
                    val c = 0.7978845608f // sqrt(2/pi)
                    val t = kotlin.math.tanh(c * inner)
                    val y = 0.5f * x * (1.0f + t)
                    @Suppress("UNCHECKED_CAST")
                    y as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for gelu: ${'$'}{tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-tril")
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> {
        val rank = tensor.rank
        // Apply over last two dims; for rank < 2, just copy
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
            if (rank < 2) {
                @Suppress("UNCHECKED_CAST")
                return@init tensor.data.get(*outIdx) as V
            }
            val rows = tensor.shape[rank - 2]
            val cols = tensor.shape[rank - 1]
            val i = outIdx[rank - 2]
            val j = outIdx[rank - 1]
            val keep = j - i <= k
            if (keep) {
                @Suppress("UNCHECKED_CAST")
                tensor.data.get(*outIdx) as V
            } else {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        @Suppress("UNCHECKED_CAST")
                        0.0f as V
                    }
                    sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                        @Suppress("UNCHECKED_CAST")
                        0 as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for tril: ${'$'}{tensor.dtype}")
                }
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sum")
    override fun <T : DType, V> sum(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        // Determine reduction mode
        floatBufferOf(tensor)?.let { src ->
            if (dim == null) {
                var acc = 0.0f
                for (i in src.indices) acc += src[i]
                return floatResult(Shape(), tensor.dtype, floatArrayOf(acc), tensor)
            }
            val nd = if (dim < 0) dim + rank else dim
            if (nd in 0 until rank) {
                val dims = tensor.shape.dimensions
                var outer = 1
                for (i in 0 until nd) outer *= dims[i]
                val red = dims[nd]
                var inner = 1
                for (i in nd + 1 until rank) inner *= dims[i]
                val out = FloatArray(outer * inner)
                for (o in 0 until outer) {
                    val srcBase = o * red * inner
                    val outBase = o * inner
                    for (k in 0 until red) {
                        val rowBase = srcBase + k * inner
                        for (x in 0 until inner) out[outBase + x] += src[rowBase + x]
                    }
                }
                val outDims = IntArray(rank - 1)
                var oi = 0
                for (i in 0 until rank) { if (i == nd) continue; outDims[oi++] = dims[i] }
                return floatResult(Shape(outDims), tensor.dtype, out, tensor)
            }
        }
        if (dim == null) {
            // Reduce all elements to a scalar (rank-0)
            val outShape = Shape()
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        var acc = 0.0f
                        val dims = tensor.shape.dimensions
                        if (dims.isEmpty()) {
                            acc = tensor.data.get() as Float
                        } else if (dims.any { it == 0 }) {
                            // Empty tensor: by convention sum over all dims is zero
                            acc = 0.0f
                        } else {
                            val idx = IntArray(dims.size) { 0 }
                            while (true) {
                                acc += tensor.data.get(*idx) as Float
                                var d = dims.size - 1
                                while (d >= 0) {
                                    idx[d]++
                                    if (idx[d] < dims[d]) break
                                    idx[d] = 0
                                    d--
                                }
                                if (d < 0) break
                            }
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        var acc = 0
                        val dims = tensor.shape.dimensions
                        if (dims.isEmpty()) {
                            acc = tensor.data.get() as Int
                        } else if (dims.any { it == 0 }) {
                            // Empty tensor: sum is zero
                            acc = 0
                        } else {
                            val idx = IntArray(dims.size) { 0 }
                            while (true) {
                                acc += tensor.data.get(*idx) as Int
                                var d = dims.size - 1
                                while (d >= 0) {
                                    idx[d]++
                                    if (idx[d] < dims[d]) break
                                    idx[d] = 0
                                    d--
                                }
                                if (d < 0) break
                            }
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for sum: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "sum: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1) { 0 }
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        var acc = 0.0f
                        val inIdx = IntArray(rank)
                        var o = 0
                        for (i in 0 until rank) {
                            if (i == nd) continue
                            inIdx[i] = outIdx[o++]
                        }
                        for (k in 0 until inDims[nd]) {
                            inIdx[nd] = k
                            acc += tensor.data.get(*inIdx) as Float
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        var acc = 0
                        val inIdx = IntArray(rank)
                        var o = 0
                        for (i in 0 until rank) {
                            if (i == nd) continue
                            inIdx[i] = outIdx[o++]
                        }
                        for (k in 0 until inDims[nd]) {
                            inIdx[nd] = k
                            acc += tensor.data.get(*inIdx) as Int
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for sum: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-argmax")
    override fun <T : DType, V> argMax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        val rank = tensor.rank
        require(rank >= 1) { "argMax: input must have rank >= 1" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "argMax: dim $dim out of range for rank $rank" }
        val inDims = tensor.shape.dimensions
        // Reduced (dim removed) shape; scalar -> Shape(1) to match VoidTensorOps.
        val reduced = IntArray(rank - 1)
        run { var o = 0; for (i in 0 until rank) if (i != nd) reduced[o++] = inDims[i] }
        val outShape = if (reduced.isEmpty()) Shape(1) else Shape(reduced)
        val outCount = reduced.fold(1) { a, b -> a * b }
        val indices = IntArray(outCount)
        val outIdx = IntArray(reduced.size)
        for (flat in 0 until outCount) {
            val inIdx = IntArray(rank)
            run { var o = 0; for (i in 0 until rank) if (i != nd) inIdx[i] = outIdx[o++] }
            var bestK = 0
            var bestV = Float.NEGATIVE_INFINITY
            for (k in 0 until inDims[nd]) {
                inIdx[nd] = k
                val v = (tensor.data.get(*inIdx) as Number).toFloat()
                if (v > bestV) { bestV = v; bestK = k } // strict > keeps the LOWEST index on ties
            }
            indices[flat] = bestK
            var d = reduced.size - 1
            while (d >= 0) { outIdx[d]++; if (outIdx[d] < reduced[d]) break; outIdx[d] = 0; d-- }
        }
        // Eager result: store the indices as index-valued floats in the INPUT dtype so the tensor is
        // a consistent Tensor<T,V> readable on every target (an Int32 payload inside a <FP32,Float>
        // tensor throws ClassCastException on Kotlin/Native + Wasm). The traced/compiled form is a
        // real i32 tensor — see VoidTensorOps.argMax + ArgMaxOperationsConverter (emits stablehlo i32).
        val floats = FloatArray(outCount) { indices[it].toFloat() }
        @Suppress("UNCHECKED_CAST")
        val outData = dataFactory.adoptFloatArray<T, Float>(outShape, tensor.dtype, floats)
            as sk.ainet.lang.tensor.data.TensorData<T, V>
        return CpuTensor(outData, this, tensor.dtype, GradState(requiresGrad = false))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-mean")
    override fun <T : DType, V> mean(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        if (tensor.volume > 0) floatBufferOf(tensor)?.let { src ->
            if (dim == null) {
                var acc = 0.0f
                for (i in src.indices) acc += src[i]
                return floatResult(Shape(), tensor.dtype, floatArrayOf(acc / tensor.volume.toFloat()), tensor)
            }
            val nd = if (dim < 0) dim + rank else dim
            if (nd in 0 until rank) {
                val dims = tensor.shape.dimensions
                var outer = 1
                for (i in 0 until nd) outer *= dims[i]
                val red = dims[nd]
                if (red > 0) {
                    var inner = 1
                    for (i in nd + 1 until rank) inner *= dims[i]
                    val out = FloatArray(outer * inner)
                    for (o in 0 until outer) {
                        val srcBase = o * red * inner
                        val outBase = o * inner
                        for (k in 0 until red) {
                            val rowBase = srcBase + k * inner
                            for (x in 0 until inner) out[outBase + x] += src[rowBase + x]
                        }
                    }
                    val invN = red.toFloat()
                    for (x in out.indices) out[x] /= invN
                    val outDims = IntArray(rank - 1)
                    var oi = 0
                    for (i in 0 until rank) { if (i == nd) continue; outDims[oi++] = dims[i] }
                    return floatResult(Shape(outDims), tensor.dtype, out, tensor)
                }
            }
        }
        if (dim == null) {
            val outShape = Shape()
            val count = if (tensor.volume == 0) 0 else tensor.volume
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (count == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var acc = 0.0f
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                acc = tensor.data.get() as Float
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    acc += tensor.data.get(*idx) as Float
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            @Suppress("UNCHECKED_CAST") (acc / count.toFloat()) as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (count == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var acc = 0
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                acc = tensor.data.get() as Int
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    acc += tensor.data.get(*idx) as Int
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            @Suppress("UNCHECKED_CAST") (acc / count) as V
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for mean: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "mean: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val n = inDims[nd]
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var acc = 0.0f
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                acc += tensor.data.get(*inIdx) as Float
                            }
                            @Suppress("UNCHECKED_CAST") (acc / n.toFloat()) as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var acc = 0
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                acc += tensor.data.get(*inIdx) as Int
                            }
                            @Suppress("UNCHECKED_CAST") (acc / n) as V
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for mean: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-variance")
    override fun <T : DType, V> variance(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        if (dim == null) {
            val outShape = Shape()
            val n = tensor.volume
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var sum = 0.0f
                            var sumSq = 0.0f
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                val x = tensor.data.get() as Float
                                sum = x
                                sumSq = x * x
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    val x = tensor.data.get(*idx) as Float
                                    sum += x
                                    sumSq += x * x
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            val mean = sum / n.toFloat()
                            val varVal = (sumSq / n.toFloat()) - mean * mean
                            @Suppress("UNCHECKED_CAST") varVal as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var sum = 0L
                            var sumSq = 0L
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                val x = (tensor.data.get() as Int).toLong()
                                sum = x
                                sumSq = x * x
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    val x = (tensor.data.get(*idx) as Int).toLong()
                                    sum += x
                                    sumSq += x * x
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            val meanNum = sum / n.toLong()
                            val varNum = (sumSq / n.toLong()) - meanNum * meanNum
                            @Suppress("UNCHECKED_CAST") (varNum.toInt() as V)
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for variance: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "variance: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val n = inDims[nd]
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var sum = 0.0f
                            var sumSq = 0.0f
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                val x = tensor.data.get(*inIdx) as Float
                                sum += x
                                sumSq += x * x
                            }
                            val mean = sum / n.toFloat()
                            val varVal = (sumSq / n.toFloat()) - mean * mean
                            @Suppress("UNCHECKED_CAST") varVal as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var sum = 0L
                            var sumSq = 0L
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                val x = (tensor.data.get(*inIdx) as Int).toLong()
                                sum += x
                                sumSq += x * x
                            }
                            val meanNum = sum / n.toLong()
                            val varNum = (sumSq / n.toLong()) - meanNum * meanNum
                            @Suppress("UNCHECKED_CAST") (varNum.toInt() as V)
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for variance: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sqrt")
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        require(
            tensor.dtype == sk.ainet.lang.types.FP32::class ||
                tensor.dtype == sk.ainet.lang.types.FP16::class
        ) { "sqrt supports only FP16/FP32, got ${tensor.dtype}" }

        floatUnaryFast(tensor) { x -> sqrt(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val v = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            sqrt(v) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    /**
     * Element-wise power: `c[i] = a[i] ^ b[i]`. Integer-valued exponents
     * use repeated multiply for stability; everything else routes through
     * `kotlin.math.pow`. Shape contract: shapes must match exactly (no
     * broadcasting yet — caller's responsibility).
     */
    override fun <T : DType, V> pow(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        require(
            a.dtype == sk.ainet.lang.types.FP32::class ||
                a.dtype == sk.ainet.lang.types.FP16::class
        ) { "pow supports only FP16/FP32, got ${a.dtype}" }
        require(a.shape == b.shape) { "pow requires matching shapes; got ${a.shape} and ${b.shape}" }
        floatBinaryFast(a, b) { x, y -> scalarPow(x, y) }?.let { return it }
        val outData = dataFactory.init<T, V>(a.shape, a.dtype) { idx ->
            val av = a.data.get(*idx) as Float
            val bv = b.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            scalarPow(av, bv) as V
        }
        return newTensor(outData, a.dtype, a)
    }

    /**
     * Element-wise scalar power: `c[i] = a[i] ^ n`. Small-integer
     * exponents (|n| <= 16) use repeated multiply for exactness; all
     * other values route through `kotlin.math.pow`.
     */
    override fun <T : DType, V> powScalar(a: Tensor<T, V>, n: Number): Tensor<T, V> {
        require(
            a.dtype == sk.ainet.lang.types.FP32::class ||
                a.dtype == sk.ainet.lang.types.FP16::class
        ) { "powScalar supports only FP16/FP32, got ${a.dtype}" }
        val nFloat = n.toFloat()
        val nInt = n.toInt()
        val isSmallInt = nFloat == nInt.toFloat() && kotlin.math.abs(nInt) <= 16
        floatUnaryFast(a) { x -> if (isSmallInt) integerPow(x, nInt) else scalarPow(x, nFloat) }?.let { return it }
        val outData = dataFactory.init<T, V>(a.shape, a.dtype) { idx ->
            val av = a.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            (if (isSmallInt) integerPow(av, nInt) else scalarPow(av, nFloat)) as V
        }
        return newTensor(outData, a.dtype, a)
    }

    /** Repeated-multiply for small integer exponents. Handles n < 0 via reciprocal. */
    private fun integerPow(base: Float, n: Int): Float {
        if (n == 0) return 1f
        if (n < 0) return 1f / integerPow(base, -n)
        var result = 1f
        var b = base
        var e = n
        while (e > 0) {
            if (e and 1 == 1) result *= b
            b *= b
            e = e ushr 1
        }
        return result
    }

    private fun scalarPow(base: Float, exp: Float): Float =
        base.toDouble().pow(exp.toDouble()).toFloat()

    /**
     * Element-wise natural log: `c[i] = ln(a[i])`. Negative or zero
     * inputs follow `kotlin.math.ln` semantics (negative → NaN, zero
     * → -Infinity). Mirror of `stablehlo.log`.
     */
    override fun <T : DType, V> log(tensor: Tensor<T, V>): Tensor<T, V> {
        require(
            tensor.dtype == sk.ainet.lang.types.FP32::class ||
                tensor.dtype == sk.ainet.lang.types.FP16::class
        ) { "log supports only FP16/FP32, got ${tensor.dtype}" }
        floatUnaryFast(tensor) { x -> ln(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val v = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            ln(v) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    /** Element-wise base-2 log: `c[i] = log2(a[i])`. */
    override fun <T : DType, V> log2(tensor: Tensor<T, V>): Tensor<T, V> {
        require(
            tensor.dtype == sk.ainet.lang.types.FP32::class ||
                tensor.dtype == sk.ainet.lang.types.FP16::class
        ) { "log2 supports only FP16/FP32, got ${tensor.dtype}" }
        floatUnaryFast(tensor) { x -> kmLog2(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val v = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kmLog2(v) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    /** Element-wise base-10 log: `c[i] = log10(a[i])`. */
    override fun <T : DType, V> log10(tensor: Tensor<T, V>): Tensor<T, V> {
        require(
            tensor.dtype == sk.ainet.lang.types.FP32::class ||
                tensor.dtype == sk.ainet.lang.types.FP16::class
        ) { "log10 supports only FP16/FP32, got ${tensor.dtype}" }
        floatUnaryFast(tensor) { x -> kmLog10(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val v = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kmLog10(v) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    // ---- TinyFoA ops: abs, sign, clamp, lt, ge ----

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-abs")
    override fun <T : DType, V> abs(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.abs(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    kotlin.math.abs(v) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    kotlin.math.abs(v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for abs: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-sign")
    override fun <T : DType, V> sign(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x > 0f) 1f else if (x < 0f) -1f else 0f }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v > 0f) 1f else if (v < 0f) -1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v > 0) 1 else if (v < 0) -1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for sign: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-clamp")
    override fun <T : DType, V> clamp(tensor: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V> {
        require(minVal <= maxVal) { "clamp: minVal ($minVal) must be <= maxVal ($maxVal)" }
        floatUnaryFast(tensor) { x -> x.coerceIn(minVal, maxVal) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    v.coerceIn(minVal, maxVal) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    v.coerceIn(minVal.toInt(), maxVal.toInt()) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for clamp: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-lt")
    override fun <T : DType, V> lt(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x < value) 1f else 0f }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < value) 1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < value.toInt()) 1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for lt: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-ge")
    override fun <T : DType, V> ge(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> if (x >= value) 1f else 0f }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= value) 1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= value.toInt()) 1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for ge: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    // ---- narrow, pad2d, unfold ----

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-narrow")
    override fun <T : DType, V> narrow(tensor: Tensor<T, V>, dim: Int, start: Int, length: Int): Tensor<T, V> {
        val actualDim = if (dim < 0) tensor.shape.rank + dim else dim
        require(actualDim in 0 until tensor.shape.rank) { "narrow dim $dim out of bounds for rank ${tensor.shape.rank}" }
        require(start >= 0 && start + length <= tensor.shape.dimensions[actualDim]) {
            "narrow: start=$start length=$length exceeds dim size ${tensor.shape.dimensions[actualDim]}"
        }
        val resultDims = tensor.shape.dimensions.copyOf()
        resultDims[actualDim] = length
        val outShape = Shape(resultDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            val srcIdx = idx.copyOf()
            srcIdx[actualDim] = srcIdx[actualDim] + start
            tensor.data.get(*srcIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-pad2d")
    override fun <T : DType, V> pad2d(tensor: Tensor<T, V>, padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V> {
        require(tensor.shape.rank == 4) { "pad2d requires 4D tensor [N,C,H,W], got rank ${tensor.shape.rank}" }
        val (n, c, h, w) = tensor.shape.dimensions.toList()
        val newH = h + padTop + padBottom
        val newW = w + padLeft + padRight
        val outShape = Shape(n, c, newH, newW)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            val srcRow = idx[2] - padTop
            val srcCol = idx[3] - padLeft
            if (srcRow in 0 until h && srcCol in 0 until w) {
                tensor.data.get(idx[0], idx[1], srcRow, srcCol)
            } else {
                // Zero padding
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        @Suppress("UNCHECKED_CAST") (0f as V)
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        @Suppress("UNCHECKED_CAST") (0 as V)
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for pad2d: ${tensor.dtype}")
                }
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-unfold")
    override fun <T : DType, V> unfold(tensor: Tensor<T, V>, dim: Int, size: Int, step: Int): Tensor<T, V> {
        val actualDim = if (dim < 0) tensor.shape.rank + dim else dim
        require(actualDim in 0 until tensor.shape.rank) { "unfold dim $dim out of bounds for rank ${tensor.shape.rank}" }
        val dimSize = tensor.shape.dimensions[actualDim]
        require(size <= dimSize) { "unfold size $size > dim size $dimSize" }
        val numWindows = (dimSize - size) / step + 1
        val resultDims = IntArray(tensor.shape.rank + 1)
        for (i in 0 until tensor.shape.rank) {
            resultDims[i] = if (i == actualDim) numWindows else tensor.shape.dimensions[i]
        }
        resultDims[tensor.shape.rank] = size
        val outShape = Shape(resultDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            // idx has rank+1 dimensions. Last dimension is the window element index.
            val windowIdx = idx[tensor.shape.rank]
            val srcIdx = IntArray(tensor.shape.rank)
            for (i in 0 until tensor.shape.rank) {
                srcIdx[i] = if (i == actualDim) {
                    idx[i] * step + windowIdx
                } else {
                    idx[i]
                }
            }
            tensor.data.get(*srcIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-convert")
    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        @Suppress("UNCHECKED_CAST")
        val targetClass = targetType::class as KClass<TTo>
        if (tensor.dtype == targetClass) {
            @Suppress("UNCHECKED_CAST")
            return tensor as Tensor<TTo, V>
        }

        @Suppress("UNCHECKED_CAST")
        val outData = when (targetClass) {
            FP32::class -> dataFactory.adoptFloatArray<TTo, Float>(
                tensor.shape,
                targetClass,
                copyTensorValuesAsFloatArray(tensor)
            ) as TensorData<TTo, V>
            // Narrowing to 16 bits must actually ROUND. This previously shared the FP32 branch,
            // so `convert(x, FP16)` only re-tagged the dtype and kept full FP32 precision — the
            // resulting tensor claimed to be FP16 while holding values no FP16 can represent, with
            // no rounding, no overflow-to-Inf past the format's range, and no NaN handling.
            // Storage stays float-backed (matching DenseTensorDataFactory's narrow tags); it is
            // the VALUES that now reflect the target format.
            FP16::class, BF16::class -> {
                val codec = if (targetClass == BF16::class) Bf16Codec else Fp16Codec
                val src = copyTensorValuesAsFloatArray(tensor)
                val rounded = FloatArray(src.size) { codec.decode(codec.encode(src[it])) }
                dataFactory.adoptFloatArray<TTo, Float>(
                    tensor.shape,
                    targetClass,
                    rounded
                ) as TensorData<TTo, V>
            }
            Int32::class, Int8::class -> dataFactory.fromIntArray<TTo, Int>(
                tensor.shape,
                targetClass,
                copyTensorValuesAsIntArray(tensor)
            ) as TensorData<TTo, V>
            else -> throw IllegalArgumentException(
                "convert supports FP32, FP16, BF16, Int32, and Int8 targets, got ${targetType.name}"
            )
        }
        return CpuTensor(outData, this, targetClass, GradState(requiresGrad = tensor.requiresGrad))
    }

    override fun <T : DType, V> gather(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        require(dim == 0) { "gather: only dim=0 supported currently, got dim=$dim" }
        // Gather rows from input along dimension 0 using indices
        // Input: [vocabSize, embeddingDim], Indices: [L] or [N, L]
        // Output: [L, embeddingDim] or [N, L, embeddingDim]
        val numIndices = indices.volume
        // Read the indices in row-major order. The vararg element accessor
        // requires one coordinate per dimension, so a flat `data[i]` throws for
        // rank >= 2 indices — read the contiguous buffer when present, otherwise
        // unravel the flat position into a coordinate.
        val idxData = indices.data
        val indexList = when (idxData) {
            is IntArrayTensorData<*> -> IntArray(numIndices) { idxData.buffer[it] }
            is FloatArrayTensorData<*> -> IntArray(numIndices) { idxData.buffer[it].toInt() }
            else -> {
                val dims = indices.shape.dimensions
                IntArray(numIndices) { flat ->
                    val coord = IntArray(dims.size)
                    var rem = flat
                    for (d in dims.indices.reversed()) {
                        coord[d] = rem % dims[d]
                        rem /= dims[d]
                    }
                    (idxData.get(*coord) as Number).toInt()
                }
            }
        }

        if (input.rank == 2) {
            val embDim = input.shape[1]
            val outShape = if (indices.rank == 1) {
                Shape(intArrayOf(numIndices, embDim))
            } else {
                // Preserve index shape + embedding dim
                Shape(IntArray(indices.rank) { indices.shape[it] } + intArrayOf(embDim))
            }
            fun rowOf(outIdx: IntArray): Int {
                // Map multi-dim output index to the flat index into the index list.
                val flatIdx = if (outIdx.size == 2) outIdx[0] else {
                    var flat = 0
                    for (d in 0 until outIdx.size - 1) {
                        flat = flat * (if (d < indices.rank) indices.shape[d] else 1) + outIdx[d]
                    }
                    flat
                }
                return indexList[flatIdx]
            }
            val src = input.data
            val outData = if (src is RowDequantSource) {
                // Packed / oversized table (e.g. a Q-quantised embedding): dequantise only the rows
                // actually touched — never materialise the whole table, never call get() (unsupported on
                // such tensors). Each unique row is dequantised once; logical dtype is FP32.
                val rowCache = HashMap<Int, FloatArray>()
                dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
                    val row = rowOf(outIdx)
                    val col = outIdx[outIdx.size - 1]
                    @Suppress("UNCHECKED_CAST")
                    (rowCache.getOrPut(row) { src.dequantRow(row) }[col] as V)
                }
            } else {
                dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
                    input.data[rowOf(outIdx), outIdx[outIdx.size - 1]]
                }
            }
            return newTensor(outData, input.dtype, input)
        }

        // Fallback for higher-rank inputs
        error("gather: unsupported input rank ${input.rank}")
    }

    override fun <T : DType, V> indexSelect(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        require(dim in 0 until input.rank) { "indexSelect: dim=$dim out of range for rank ${input.rank}" }
        val numIndices = indices.volume
        // The vararg element accessor requires one coordinate per dimension, so a flat
        // `data[i]` throws for rank >= 2 indices — read the contiguous buffer when present,
        // otherwise unravel the flat position into a coordinate (mirrors gather() above).
        val idxData = indices.data
        val indexList = when (idxData) {
            is IntArrayTensorData<*> -> IntArray(numIndices) { idxData.buffer[it] }
            is FloatArrayTensorData<*> -> IntArray(numIndices) { idxData.buffer[it].toInt() }
            else -> {
                val dims = indices.shape.dimensions
                IntArray(numIndices) { flat ->
                    val coord = IntArray(dims.size)
                    var rem = flat
                    for (d in dims.indices.reversed()) {
                        coord[d] = rem % dims[d]
                        rem /= dims[d]
                    }
                    (idxData.get(*coord) as Number).toInt()
                }
            }
        }

        val resultDims = input.shape.dimensions.copyOf()
        resultDims[dim] = numIndices
        val resultShape = Shape(resultDims)

        val outData = dataFactory.init<T, V>(resultShape, input.dtype) { outIdx ->
            // Replace the dim-th index with the looked-up value from indexList
            val srcIdx = outIdx.copyOf()
            srcIdx[dim] = indexList[outIdx[dim]]
            input.data.get(*srcIdx)
        }
        return newTensor(outData, input.dtype, input)
    }

    override fun <T : DType, V> exp(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.exp(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.exp(x) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> expm1(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.expm1(x.toDouble()).toFloat() }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.expm1(x.toDouble()).toFloat() as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> sin(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.sin(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.sin(x) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> cos(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.cos(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.cos(x) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-tanh")
    override fun <T : DType, V> tanh(tensor: Tensor<T, V>): Tensor<T, V> {
        floatUnaryFast(tensor) { x -> kotlin.math.tanh(x) }?.let { return it }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    kotlin.math.tanh(x) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for tanh: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> scaledDotProductAttention(
        query: Tensor<T, V>,
        key: Tensor<T, V>,
        value: Tensor<T, V>,
        mask: Tensor<T, V>?,
        scale: Float,
        causal: Boolean
    ): Tensor<T, V> {
        // Expected shapes: [batch, heads, seqQ, headDim] for Q, [batch, heads, seqKV, headDim] for K/V
        require(query.rank == 4) { "SDPA: expected rank-4 query, got rank ${query.rank}" }
        require(key.rank == 4) { "SDPA: expected rank-4 key, got rank ${key.rank}" }
        require(value.rank == 4) { "SDPA: expected rank-4 value, got rank ${value.rank}" }
        require(query.shape[0] == key.shape[0] && query.shape[0] == value.shape[0]) {
            "SDPA: batch mismatch — Q=${query.shape[0]} K=${key.shape[0]} V=${value.shape[0]}"
        }
        require(query.shape[1] == key.shape[1] && query.shape[1] == value.shape[1]) {
            "SDPA: head count mismatch — Q=${query.shape[1]} K=${key.shape[1]} V=${value.shape[1]}. " +
                "For grouped-query attention, K/V must be tiled to Q's head count upstream."
        }
        require(query.shape[3] == key.shape[3]) {
            "SDPA: Q head_dim (${query.shape[3]}) does not match K head_dim (${key.shape[3]})"
        }
        require(query.shape[3] == value.shape[3]) {
            "SDPA: Q head_dim (${query.shape[3]}) does not match V head_dim (${value.shape[3]})"
        }
        require(key.shape[2] == value.shape[2]) {
            "SDPA: K seqKV (${key.shape[2]}) does not match V seqKV (${value.shape[2]})"
        }

        val batch = query.shape[0]
        val heads = query.shape[1]
        val seqQ = query.shape[2]
        val headDim = query.shape[3]
        val seqKV = key.shape[2]

        // The signature default `scale = 0f` means "use the standard
        // 1/sqrt(headDim)"; applying 0 literally would flatten every softmax to
        // a uniform distribution. Resolve it here.
        val effectiveScale = if (scale == 0f) (1.0 / kotlin.math.sqrt(headDim.toDouble())).toFloat() else scale

        val qBuf = query.data.copyToFloatArray()
        val kBuf = key.data.copyToFloatArray()
        val vBuf = value.data.copyToFloatArray()
        // Hoisted out of the per-head loop: the mask is read-only for the whole call.
        val maskBuf = mask?.data?.copyToFloatArray()

        val outBuf = FloatArray(batch * heads * seqQ * headDim)

        // SKEEP-005: every (batch, head) pair is independent — private scores scratch, disjoint
        // output rows — so the pairs are the schedule's units. The per-pair arithmetic and its
        // order are exactly the sequential loop's, which keeps a scheduled run bit-identical.
        // Tiny calls (decode steps on a handful of heads) stay on the caller: a region costs more
        // than it saves below SDPA_PARALLEL_MIN_WORK multiply-adds.
        val units = batch * heads
        val workPerUnit = seqQ.toLong() * seqKV.toLong() * headDim.toLong() * 2L
        val regionSchedule = if (workPerUnit * units < SDPA_PARALLEL_MIN_WORK) sk.ainet.context.schedule.Schedule.Sequential else schedule
        regionSchedule.forRange(units, grain = 1) { unitStart, unitEnd ->
            // Per-task scratch: a plain heap array, never the context's scratch pool or step slab
            // (both single-threaded). Every entry is overwritten by the QK^T pass, so one array
            // serves every unit of this task.
            val scores = FloatArray(seqQ * seqKV)
            for (unit in unitStart until unitEnd) {
                val b = unit / heads
                val h = unit % heads
                // Compute attention scores: Q @ K^T, then scale
                for (qi in 0 until seqQ) {
                    for (ki in 0 until seqKV) {
                        var dot = 0f
                        val qOff = ((b * heads + h) * seqQ + qi) * headDim
                        val kOff = ((b * heads + h) * seqKV + ki) * headDim
                        for (d in 0 until headDim) {
                            dot += qBuf[qOff + d] * kBuf[kOff + d]
                        }
                        scores[qi * seqKV + ki] = dot * effectiveScale
                    }
                }

                // Apply causal mask (set future positions to -inf)
                if (causal) {
                    for (qi in 0 until seqQ) {
                        // For single-token inference with KV cache: qi indexes from the
                        // query's perspective. The valid range of ki is [0, seqKV).
                        // With cache, seqQ=1 and seqKV=pos+1, so all keys are valid (no masking needed).
                        // Without cache, seqQ=seqKV, and we mask ki > qi.
                        val maxKi = if (seqQ == seqKV) qi else seqKV - seqQ + qi
                        for (ki in maxKi + 1 until seqKV) {
                            scores[qi * seqKV + ki] = Float.NEGATIVE_INFINITY
                        }
                    }
                }

                // Apply external mask if provided
                if (maskBuf != null) {
                    for (i in scores.indices) {
                        scores[i] += maskBuf[i % maskBuf.size]
                    }
                }

                // Softmax over the key dimension for each query position
                for (qi in 0 until seqQ) {
                    val off = qi * seqKV
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (ki in 0 until seqKV) {
                        if (scores[off + ki] > maxVal) maxVal = scores[off + ki]
                    }
                    var sumExp = 0f
                    for (ki in 0 until seqKV) {
                        val e = kotlin.math.exp(scores[off + ki] - maxVal)
                        scores[off + ki] = e
                        sumExp += e
                    }
                    if (sumExp > 0f) {
                        for (ki in 0 until seqKV) {
                            scores[off + ki] /= sumExp
                        }
                    }
                }

                // Compute output: scores @ V
                for (qi in 0 until seqQ) {
                    val outOff = ((b * heads + h) * seqQ + qi) * headDim
                    for (d in 0 until headDim) {
                        var sum = 0f
                        for (ki in 0 until seqKV) {
                            val vOff = ((b * heads + h) * seqKV + ki) * headDim
                            sum += scores[qi * seqKV + ki] * vBuf[vOff + d]
                        }
                        outBuf[outOff + d] = sum
                    }
                }
            }
        }

        val shape = Shape(batch, heads, seqQ, headDim)
        @Suppress("UNCHECKED_CAST")
        val data = dataFactory.adoptFloatArray<T, Float>(shape, query.dtype, outBuf) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(data, query.dtype, query, key, value)
    }

}

public class DefaultCpuOps(
    dataFactory: TensorDataFactory,
    schedule: sk.ainet.context.schedule.Schedule,
) : DefaultCpuOpsBase(dataFactory, schedule) {
    public constructor(dataFactory: TensorDataFactory) : this(dataFactory, sk.ainet.context.schedule.Schedule.Sequential)
}
