package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.VectorOperators
import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelServiceLoader
import sk.ainet.backend.api.kernel.KernelStrictness
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarFp16MatmulKernel
import sk.ainet.exec.kernel.ScalarMatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_0MatmulKernel
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.MemorySegmentBackedData
import sk.ainet.lang.tensor.data.MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Bf16TensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.tensor.data.Q4_0TensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.data.TensorDataFactory
import java.lang.foreign.Arena
import kotlin.math.max

internal class DefaultCpuOpsJvm(
    dataFactory: TensorDataFactory,
    schedule: sk.ainet.context.schedule.Schedule = sk.ainet.context.schedule.Schedule.Sequential,
) : DefaultCpuOpsBase(dataFactory, schedule) {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    /**
     * On the JVM, auto-install ServiceLoader-discovered providers (Panama Vector,
     * native FFM) so the base class's platform-neutral packed-quant dispatch
     * (`chooseQuantizedMatmulHeap`, used for Q5_1/Q5_0 and the non-JVM path) resolves
     * the SIMD/FFM kernels rather than only the scalar floor.
     */
    override fun ensureKernelProviders() {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
    }

    /**
     * FP32 matmul kernel resolved via [KernelRegistry]. First access on a
     * given instance auto-installs providers via [KernelServiceLoader]
     * if the registry is empty; subsequent calls reuse the cached
     * lookup. Apps that prefer to wire their own providers can call
     * `KernelRegistry.register(...)` before constructing this op set.
     * Falls back to [ScalarMatmulKernel] only when no provider reports
     * itself available — in practice, [PanamaVectorKernelProvider]
     * (priority 50) wins on JDK 21+ with the incubator module loaded.
     */
    private val fp32MatmulKernel: Fp32MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.bestAvailable()?.matmulFp32() ?: ScalarMatmulKernel
    }

    /**
     * Q4_K kernel resolved via [KernelRegistry], lazily initialized on
     * first quantized matmul call. Auto-installs ServiceLoader-discovered
     * providers when the registry is empty. Returns `null` if no
     * provider carries a Q4_K kernel — caller falls back to
     * [JvmQuantizedVectorKernels.matmulQ4_KVec], so this PR introduces
     * zero functional regression even when the SPI doesn't resolve.
     */
    private val q4kMatmulKernel: Q4KMatmulKernel? by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulQ4K() != null }
            ?.matmulQ4K()
    }

    /**
     * Q8_0 kernel resolved via [KernelRegistry], lazily initialized on
     * first quantized matmul call. Mirrors [q4kMatmulKernel] — auto-
     * installs ServiceLoader-discovered providers when the registry is
     * empty, returns `null` if no provider carries a Q8_0 kernel.
     * Caller falls back to [JvmQuantizedVectorKernels.matmulQ8_0Vec],
     * preserving the legacy code path when the SPI doesn't resolve.
     */
    private val q8_0MatmulKernel: Q8_0MatmulKernel? by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulQ8_0() != null }
            ?.matmulQ8_0()
    }

    /**
     * BF16 matmul kernel resolved via [KernelRegistry]. Unlike the Q4_K
     * and Q8_0 lookups (nullable, with legacy `JvmQuantizedVectorKernels`
     * fallbacks), BF16 has no pre-SPI implementation in this codebase —
     * the scalar SPI kernel is the floor. We mirror [fp32MatmulKernel]'s
     * pattern: non-null, picks the highest-priority provider that carries
     * a BF16 kernel (native FFM at 100, Panama Vector at 50), falls back
     * to [ScalarBf16MatmulKernel] when no SIMD provider reports
     * availability (e.g. tests that explicitly clear the registry).
     */
    private val bf16MatmulKernel: Bf16MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulBf16() != null }
            ?.matmulBf16()
            ?: ScalarBf16MatmulKernel
    }

    /** FP16 matmul kernel resolved via [KernelRegistry]; mirrors [bf16MatmulKernel]. */
    private val fp16MatmulKernel: Fp16MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulFp16() != null }
            ?.matmulFp16()
            ?: ScalarFp16MatmulKernel
    }

    /**
     * Q4_0 matmul kernel resolved via [KernelRegistry]. Mirrors
     * [bf16MatmulKernel]: non-null, picks the highest-priority provider
     * that carries a Q4_0 kernel (native FFM at 100, Panama Vector at
     * 50), falling back to [ScalarQ4_0MatmulKernel] — the scalar SPI
     * kernel is the floor (every `KernelProvider` carries one), so Q4_0
     * has no pre-SPI legacy fallback to thread through.
     */
    private val q4_0MatmulKernel: Q4_0MatmulKernel by lazy {
        if (KernelRegistry.providers().isEmpty()) {
            KernelServiceLoader.installAll()
        }
        KernelRegistry.providers()
            .firstOrNull { it.isAvailable() && it.matmulQ4_0() != null }
            ?.matmulQ4_0()
            ?: ScalarQ4_0MatmulKernel
    }

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.add(y) }) { x, y -> x + y }?.let { return it }
        return super.add(a, b)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.sub(y) }) { x, y -> x - y }?.let { return it }
        return super.subtract(a, b)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.mul(y) }) { x, y -> x * y }?.let { return it }
        return super.multiply(a, b)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.div(y) }) { x, y -> x / y }?.let { return it }
        return super.divide(a, b)
    }

    /**
     * `x · Wᵀ` with the weight relayouted once into the order this backend's packed kernels read
     * (#1096), which is the fast path the common implementation cannot take.
     *
     * `DefaultCpuOpsBase` decodes the weight where it lies instead, because it has no kernel that
     * addresses `packedData` in feed order and relayouting for one that does not exist produced a
     * tensor nothing could decode correctly (#1124). Here the kernels do exist, so the relayout
     * pays: once per weight, cached by the identity of its bytes.
     */
    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> matmulWeightTransposed(x: Tensor<T, V>, weight: Tensor<T, V>): Tensor<T, V> {
        if (weight.shape.rank != 2 || !isHeapPackedWeightForJvm(weight.data)) {
            // Packed weights OUTSIDE the JVM relayout list (the ternary formats, #1136) go to the
            // common views→KernelDispatch path — whose kernels and reference are commonMain and
            // cannot read a MemorySegment-backed activation (SKEEP-004 forbids element access over
            // SegmentStorage by design). On this tier the bridge is trivial: bulk-copy the
            // activation to the heap once per call — decode-step activations are k floats, not
            // weights — and hand the common path heap views.
            // Dense weights need this bridge just as much as packed ones. The vectorized FP32
            // kernels take heap `FloatArray` operands only, so a segment-backed activation sends
            // the whole projection to the decoding reference kernel — and a decode-step activation
            // is k floats, so the copy is trivial beside the matmul it unlocks.
            segmentActivationToHeap(x)?.let { return super.matmulWeightTransposed(it, weight) }
            return super.matmulWeightTransposed(x, weight)
        }
        // Already in feed order — the loader produced it that way (#1120). Nothing to permute and
        // nothing to cache: the kernels want the other shape label over the same bytes, which costs
        // an object rather than a copy of the weight.
        rewrapFeedOrderWeight(weight)?.let { return matmul(x, it) }
        val packed = weight.data as sk.ainet.lang.tensor.storage.PackedBlockStorage
        val source = packed.packedData
        val cached = prepackedWeightsJvm.firstOrNull { it.first === source }?.second
        val kernelOrder = cached ?: relayoutPackedWeightForKernels(weight).also { relayouted ->
            if (prepackedWeightsJvm.size >= PREPACK_CACHE_LIMIT_JVM) prepackedWeightsJvm.removeAt(0)
            prepackedWeightsJvm.add(Pair(source, relayouted as Tensor<*, *>))
        }
        return matmul(x, kernelOrder as Tensor<T, V>)
    }

    /**
     * A heap FP32 copy of an activation whose bytes live off-heap, or null when [x] is not that
     * shape. Two producers exist on this tier: TensorData that is itself segment-backed
     * ([MemorySegmentBackedData]), and TensorData whose VIEW presents a
     * [sk.ainet.lang.memory.SegmentStorage] (SKEEP-004 off-heap scope allocations).
     */
    private fun <T : DType, V> segmentActivationToHeap(x: Tensor<T, V>): Tensor<T, V>? {
        if (x.dtype != FP32::class) return null
        val count = x.shape.volume
        val arr = FloatArray(count)
        val src = x.data as? MemorySegmentBackedData
        if (src != null) {
            java.lang.foreign.MemorySegment.copy(
                src.segment, java.lang.foreign.ValueLayout.JAVA_FLOAT, src.segmentByteOffset, arr, 0, count,
            )
        } else {
            val view = x.data.view ?: return null
            if (!view.isContiguous) return null
            val seg = view.storage as? sk.ainet.lang.memory.SegmentStorage ?: return null
            java.lang.foreign.MemorySegment.copy(
                seg.segment(), java.lang.foreign.ValueLayout.JAVA_FLOAT,
                view.layout.offsetElements * java.lang.Float.BYTES.toLong(), arr, 0, count,
            )
        }
        // NOT dataFactory.adoptFloatArray: this JVM factory adopts float arrays back INTO
        // MemorySegments — the round-trip that defeats the whole copy. The base heap class is
        // what the common dispatch path can read.
        @Suppress("UNCHECKED_CAST")
        val data = sk.ainet.lang.tensor.data.DenseFloatArrayTensorData<T>(x.shape, arr) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(data, x.dtype, x)
    }

    /** Relayouted weights keyed by the identity of the bytes they came from (#1096). */
    private val prepackedWeightsJvm: MutableList<Pair<ByteArray, Tensor<*, *>>> = mutableListOf()
    private val PREPACK_CACHE_LIMIT_JVM: Int = 64

    /**
     * Below this many multiply-accumulates a dense FP32 matmul is faster done directly than handed
     * to the tile-blocked SPI kernel, whose per-call setup dominates at these sizes. Sized to cover
     * decode-step projections (m=1) while leaving prefill batches and the big vocab matmuls to the
     * kernel — at m=32 the kernel is already ~16 GFLOP/s and pulling away.
     */
    private val SMALL_FP32_MATMUL_WORK: Long = 4_000_000L

    /** The heap packed types whose JVM kernels read input-block-major bytes. */
    private fun isHeapPackedWeightForJvm(data: sk.ainet.lang.tensor.data.TensorData<*, *>): Boolean =
        data is sk.ainet.lang.tensor.data.Q4_KTensorData || data is sk.ainet.lang.tensor.data.Q5_KTensorData ||
            data is sk.ainet.lang.tensor.data.Q6_KTensorData || data is sk.ainet.lang.tensor.data.Q5_1TensorData ||
            data is sk.ainet.lang.tensor.data.Q5_0TensorData || data is sk.ainet.lang.tensor.data.Q8_0TensorData ||
            data is sk.ainet.lang.tensor.data.Q4_0TensorData

    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        // `x · Wᵀ` written as two steps (#1108) — before everything, including the strictness check
        // below, which would otherwise report a missing kernel for an operand that has a perfectly
        // good one behind the marker.
        untransposedWeight(b)?.let { return matmulWeightTransposed(a, it) }
        // Try quantized matmul path first (FP32 input x quantized weights)
        chooseQuantizedMatmul(a, b)?.let { return it }
        // Fallback to standard FP32 matmul
        chooseMatmul(a, b)?.let { return it }
        // RFC fail-fast point: if `-Dskainet.strict.kernels=true`, surface
        // the missing kernel here rather than letting `super.matmul` quietly
        // pick the scalar dequant + FP32 fallback. The strictness check is
        // a no-op when the property is unset, preserving the existing
        // adaptive behaviour.
        KernelStrictness.failIfStrict {
            val inDt = a.dtype.simpleName ?: a.dtype.toString()
            val wDt = b.dtype.simpleName ?: b.dtype.toString()
            val providers = KernelRegistry.providers().joinToString { p ->
                "${p.name}(priority=${p.priority}, available=${p.isAvailable()})"
            }.ifEmpty { "<none>" }
            "matmul ($inDt × $wDt) has no SPI kernel; would silently fall back " +
                "to super.matmul. Registered providers: $providers"
        }
        return super.matmul(a, b)
    }

    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val rank = tensor.shape.rank
        if (rank == 2) {
            val rows = tensor.shape[0]
            val cols = tensor.shape[1]
            val data = tensor.data
            // Lazy transpose for Q4/Q8 MemorySegment data: swap shape, keep data.
            // The quantized matmul kernel accesses the segment directly with
            // (inputDim, outputDim) parameters derived from the transposed shape,
            // so physical data reordering is not needed.
            if (data is Q4MemorySegmentMarker) {
                val td = data as Q4MemorySegmentTensorData
                val transposed = Q4MemorySegmentTensorData(Shape(cols, rows), td.segment, td.segmentByteOffset)
                @Suppress("UNCHECKED_CAST")
                return newTensor(transposed as TensorData<T, V>, tensor.dtype, tensor)
            }
            if (data is Q8MemorySegmentMarker) {
                val td = data as Q8MemorySegmentTensorData
                val transposed = Q8MemorySegmentTensorData(Shape(cols, rows), td.segment, td.segmentByteOffset)
                @Suppress("UNCHECKED_CAST")
                return newTensor(transposed as TensorData<T, V>, tensor.dtype, tensor)
            }
            // Q4_K (ByteArray-backed `Q4_KBlockTensorData`) transpose used to be
            // intercepted here with a bare shape swap, on the claim that
            // `JvmQuantizedVectorKernels.matmulQ4_KVec`'s `(blockIdx * outputDim + o)`
            // byte addressing matches the packed layout "as-is". That's only true
            // when there's a single quant block per row; for any wider row it silently
            // fed the kernel physically wrong bytes — the same root cause as the
            // Q5_0/Q5_1/Q8_0/Q4_0/Q5_K/Q6_K bug fixed in `DefaultCpuOpsBase.transpose`
            // (see `transposePackedBlocks`, SKaiNET-transformers#307). Falling through
            // to `super.transpose()` picks up the corrected, physically-reordering
            // implementation instead of duplicating (and re-diverging from) it here.
            //
            // Narrow-float input-major lazy transpose is handled in DefaultCpuOpsBase too —
            // nothing above intercepts it, so it falls through. Issue #888.
            // Q6_K / Q5_1 / Q5_0 / Q4_K / Q8_0 / Q4_0 transpose is handled in
            // DefaultCpuOpsBase (block-major, shared with Native); the JVM ops don't
            // intercept them here.
            // MemorySegment FP32 fast path: physical transpose via SIMD.
            // Uses Arena.ofAuto() so the result segment is reclaimed by GC
            // when the wrapping Tensor is no longer reachable. Earlier
            // ofConfined() builds leaked an arena per call, blowing 32+ GiB
            // of direct memory in inference loops (every layer × every
            // forward pass).
            if (data is MemorySegmentBackedData) {
                val arena = Arena.ofAuto()
                val result = MemorySegmentTensorData<T>(Shape(cols, rows), arena)
                val src = data as MemorySegmentBackedData
                val floatLayout = java.lang.foreign.ValueLayout.JAVA_FLOAT
                // Bulk-load source into FloatArray, transpose via tight scalar
                // loop (JIT auto-vectorizes), bulk-write destination. Replaces
                // O(rows*cols) per-element VarHandle.get/set which dominated
                // attention-path transposes.
                val srcArr = FloatArray(rows * cols)
                java.lang.foreign.MemorySegment.copy(src.segment, floatLayout, src.segmentByteOffset, srcArr, 0, rows * cols)
                val dstArr = FloatArray(rows * cols)
                for (r in 0 until rows) {
                    val rowBase = r * cols
                    for (c in 0 until cols) {
                        dstArr[c * rows + r] = srcArr[rowBase + c]
                    }
                }
                java.lang.foreign.MemorySegment.copy(dstArr, 0, result.segment, floatLayout, result.segmentByteOffset, rows * cols)
                @Suppress("UNCHECKED_CAST")
                return newTensor(result as TensorData<T, V>, tensor.dtype, tensor)
            }
        }
        return super.transpose(tensor)
    }

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Try vectorized path for FP32
        chooseConv2d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv2d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null

        val inputData = input.data as? FloatArrayTensorData<*> ?: return null
        val weightData = weight.data as? FloatArrayTensorData<*> ?: return null
        val biasData = bias?.data as? FloatArrayTensorData<*>

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]

        val cOut = weight.shape[0]
        val kH = weight.shape[2]
        val kW = weight.shape[3]

        val (sH, sW) = stride
        val (pH, pW) = padding
        val (dH, dW) = dilation

        val outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
        val outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

        if (outH <= 0 || outW <= 0) return null

        val outBuffer = FloatArray(n * cOut * outH * outW)
        val biasBuffer = biasData?.buffer

        when {
            // Grouped convolution (includes depthwise when groups == cIn == cOut)
            groups > 1 -> {
                JvmVectorKernels.conv2dGrouped(
                    inputData.buffer, weightData.buffer, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW, dH, dW,
                    groups,
                    outH, outW,
                    outBuffer
                )
            }
            // 1x1 convolution without dilation - use optimized kernel
            kH == 1 && kW == 1 && sH == 1 && sW == 1 && pH == 0 && pW == 0 && dH == 1 && dW == 1 -> {
                JvmVectorKernels.conv2d1x1Optimized(
                    inputData.buffer, weightData.buffer, biasBuffer,
                    n, cIn, inH, inW, cOut,
                    outBuffer
                )
            }
            // Standard convolution with dilation
            dH != 1 || dW != 1 -> {
                JvmVectorKernels.conv2dIm2ColDilated(
                    inputData.buffer, weightData.buffer, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW, dH, dW,
                    outH, outW,
                    outBuffer
                )
            }
            // Standard convolution without dilation
            else -> {
                JvmVectorKernels.conv2dIm2Col(
                    inputData.buffer, weightData.buffer, biasBuffer,
                    n, cIn, inH, inW,
                    cOut, kH, kW,
                    sH, sW, pH, pW,
                    outH, outW,
                    outBuffer
                )
            }
        }

        return floatResult(Shape(n, cOut, outH, outW), input.dtype, outBuffer)
    }

    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        chooseConv1d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv1d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null
        if (groups != 1) return null  // TODO: Add grouped conv1d support

        val inputData = input.data as? FloatArrayTensorData<*> ?: return null
        val weightData = weight.data as? FloatArrayTensorData<*> ?: return null
        val biasData = bias?.data as? FloatArrayTensorData<*>

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inL = input.shape[2]

        val cOut = weight.shape[0]
        val kL = weight.shape[2]

        val outL = (inL + 2 * padding - dilation * (kL - 1) - 1) / stride + 1
        if (outL <= 0) return null

        val outBuffer = FloatArray(n * cOut * outL)

        JvmVectorKernels.conv1dIm2Col(
            inputData.buffer, weightData.buffer, biasData?.buffer,
            n, cIn, inL,
            cOut, kL,
            stride, padding, dilation,
            outL,
            outBuffer
        )

        return floatResult(Shape(n, cOut, outL), input.dtype, outBuffer)
    }

    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        chooseConv3d(input, weight, bias, stride, padding, dilation, groups)?.let { return it }
        return super.conv3d(input, weight, bias, stride, padding, dilation, groups)
    }

    private fun <T : DType, V> chooseConv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V>? {
        if (input.dtype != FP32::class) return null
        if (groups != 1) return null  // TODO: Add grouped conv3d support

        val inputData = input.data as? FloatArrayTensorData<*> ?: return null
        val weightData = weight.data as? FloatArrayTensorData<*> ?: return null
        val biasData = bias?.data as? FloatArrayTensorData<*>

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inD = input.shape[2]
        val inH = input.shape[3]
        val inW = input.shape[4]

        val cOut = weight.shape[0]
        val kD = weight.shape[2]
        val kH = weight.shape[3]
        val kW = weight.shape[4]

        val (sD, sH, sW) = stride
        val (pD, pH, pW) = padding
        val (dD, dH, dW) = dilation

        val outD = (inD + 2 * pD - dD * (kD - 1) - 1) / sD + 1
        val outH = (inH + 2 * pH - dH * (kH - 1) - 1) / sH + 1
        val outW = (inW + 2 * pW - dW * (kW - 1) - 1) / sW + 1

        if (outD <= 0 || outH <= 0 || outW <= 0) return null

        val outBuffer = FloatArray(n * cOut * outD * outH * outW)

        JvmVectorKernels.conv3dIm2Col(
            inputData.buffer, weightData.buffer, biasData?.buffer,
            n, cIn, inD, inH, inW,
            cOut, kD, kH, kW,
            sD, sH, sW,
            pD, pH, pW,
            dD, dH, dW,
            outD, outH, outW,
            outBuffer
        )

        return floatResult(Shape(n, cOut, outD, outH, outW), input.dtype, outBuffer)
    }

    /**
     * Dispatch to vectorized quantized matmul kernels for Q8_0 and Q4_K weights.
     * Input must be FP32, weights can be Q8_0TensorData or Q4_KTensorData.
     * Also supports MemorySegment-backed quantized tensor data.
     */
    private fun <T : DType, V> chooseQuantizedMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        // Input must be FP32
        if (a.dtype != FP32::class) return null
        if (b.shape.rank != 2) return null
        if (a.shape.rank < 1) return null
        if (a.shape.rank == 2) return chooseQuantizedMatmul2D(a, b)

        // Defensive: attention linear projections can in principle pass `[..., in]` (see
        // linearProject's kdoc) — flatten any leading batch/sequence dims into one so the
        // specialized quant kernels below (which only understand `[batch, in]`) still get used.
        // Also covers rank-1 activations: a single-token hidden-state vector during incremental
        // decode (post-prefill, once the KV cache is warm) — `leading` is then empty and
        // `flatBatch` is 1, so `[in]` promotes to `[1, in]` and the result squeezes back down to
        // `[out]`, same as `rank > 2` already did. Previously excluded by `a.shape.rank < 2`,
        // which sent every such decode step straight to matmulGeneric's untyped per-element
        // `TensorData.get()` path — for a pre-transposed packed-quant weight (e.g.
        // PreTransposedQ4_K) that returns the raw packed byte, not a dequantized Float.
        val leading = a.shape.dimensions.copyOf(a.shape.rank - 1)
        val flatBatch = leading.fold(1) { acc, d -> acc * d }
        val inputDim = a.shape.dimensions.last()
        val a2d = reshape(a, Shape(intArrayOf(flatBatch, inputDim)))
        val result2d = chooseQuantizedMatmul2D(a2d, b) ?: return null
        val outputDim = result2d.shape.dimensions.last()
        return reshape(result2d, Shape(leading + outputDim))
    }

    private fun <T : DType, V> chooseQuantizedMatmul2D(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        val bData = b.data
        val bShape = b.shape

        val batchSize = a.shape[0]
        val inputDim = a.shape[1]
        val weightInputDim = bShape[0]
        val outputDim = bShape[1]

        if (inputDim != weightInputDim) return null

        // Fast path: MemorySegment input × Q8 MemorySegment weights — avoid heap copy entirely
        val aData = a.data
        if (aData is MemorySegmentBackedData && bData is Q8MemorySegmentMarker) {
            return chooseQ8MemSegInputMatmul(aData, bData, batchSize, inputDim, outputDim, a)
        }

        // Get input as FloatArray — works for both FloatArrayTensorData and MemorySegmentTensorData
        val inputBuffer: FloatArray = when (aData) {
            is FloatArrayTensorData<*> -> aData.buffer
            is MemorySegmentBackedData -> aData.copyToFloatArray()
            // Any other *dense* FP32 activation — a slab-backed StorageFloatTensorData from a
            // ScopedExecutionContext forward (#1145/#1146), a view-only wrapper — copies out
            // through its own (offset-aware) copyToFloatArray. Bailing out here instead used to
            // drop these activations to matmulGeneric, whose per-element get() on a quantized
            // weight returns raw codes, not values — silently wrong logits, the #993 class of bug.
            // A packed/encoded *activation* (encoding != null) still bails: this chooser's
            // kernels want dense FP32 input.
            else -> if (aData.encoding == null) aData.copyToFloatArray() else return null
        }

        return when (bData) {
            is Q8_0TensorData -> {
                val outBuffer = FloatArray(batchSize * outputDim)
                val spiKernel = q8_0MatmulKernel
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    if (spiKernel != null) {
                        spiKernel.matmul(
                            batchInput, 0,
                            bData.packedData, 0,
                            inputDim, outputDim,
                            outBuffer, batch * outputDim,
                        )
                    } else {
                        JvmQuantizedVectorKernels.matmulQ8_0Vec(
                            batchInput,
                            bData.packedData,
                            inputDim,
                            outputDim,
                            outBuffer,
                            batch * outputDim,
                            schedule,
                        )
                    }
                }
                floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
            }
            is Q4_0TensorData -> {
                val outBuffer = FloatArray(batchSize * outputDim)
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    q4_0MatmulKernel.matmul(
                        batchInput, 0,
                        bData.packedData, 0,
                        inputDim, outputDim,
                        outBuffer, batch * outputDim,
                    )
                }
                floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
            }
            // Q5_1 / Q5_0 dispatch is handled in DefaultCpuOpsBase via the kernel
            // registry (block-major, shared with Native); not intercepted here.
            is Q4_KTensorData -> {
                val outBuffer = FloatArray(batchSize * outputDim)
                val spiKernel = q4kMatmulKernel
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    if (spiKernel != null) {
                        spiKernel.matmul(
                            batchInput, 0,
                            bData.packedData, 0,
                            inputDim, outputDim,
                            outBuffer, batch * outputDim,
                            schedule,
                        )
                    } else {
                        JvmQuantizedVectorKernels.matmulQ4_KVec(
                            batchInput,
                            bData.packedData,
                            inputDim,
                            outputDim,
                            outBuffer,
                            batch * outputDim,
                            schedule,
                        )
                    }
                }
                floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
            }
            is NarrowFloatTensorData -> {
                // Narrow floats are dense (not block-quantized) and the kernel SPI is a
                // full SGEMM with `(m, n, k)` strides — no per-batch loop needed,
                // unlike the matvec-shaped Q4_K / Q8_0 / Q6_K branches.
                //
                // The codec selects the kernel: the two formats have different bit layouts,
                // so decoding F16 bytes with the BF16 kernel would yield silent garbage.
                val kernel = when (bData.codec) {
                    Bf16Codec -> bf16MatmulKernel
                    Fp16Codec -> fp16MatmulKernel
                    else -> null
                }
                if (kernel == null) {
                    null
                } else {
                    val outBuffer = FloatArray(batchSize * outputDim)
                    kernel.matmul(
                        inputBuffer, 0, inputDim,
                        bData.packedData, 0, outputDim * NarrowFloatTensorData.BYTES_PER_ELEMENT,
                        outBuffer, 0, outputDim,
                        batchSize, outputDim, inputDim,
                    )
                    floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
                }
            }
            // Q6_K / Q5_1 / Q5_0 dispatch is handled in DefaultCpuOpsBase via the kernel
            // registry (block-major, shared with Native); not intercepted here.
            // MemorySegment-backed quantized weights (Q4/Q8) — dispatch to MemorySegment kernels
            is MemorySegmentBackedData -> {
                chooseQuantizedMatmulMemSeg(inputBuffer, bData, batchSize, inputDim, outputDim, a)
            }
            else -> null
        }
    }

    /**
     * Dispatch quantized matmul when weight data is MemorySegment-backed.
     * The concrete type is checked via marker interfaces set during weight loading.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> chooseQuantizedMatmulMemSeg(
        inputBuffer: FloatArray,
        bData: MemorySegmentBackedData,
        batchSize: Int,
        inputDim: Int,
        outputDim: Int,
        a: Tensor<T, V>,
    ): Tensor<T, V>? {
        val outBuffer = FloatArray(batchSize * outputDim)
        when (bData) {
            is Q4MemorySegmentMarker -> {
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    JvmQuantizedVectorKernels.matmulF32Q4_0MemSeg(
                        batchInput,
                        bData.segment,
                        bData.segmentByteOffset,
                        inputDim,
                        outputDim,
                        outBuffer,
                        batch * outputDim,
                    )
                }
            }
            is Q8MemorySegmentMarker -> {
                for (batch in 0 until batchSize) {
                    val batchInput = if (batchSize == 1) inputBuffer
                    else inputBuffer.copyOfRange(batch * inputDim, (batch + 1) * inputDim)
                    JvmQuantizedVectorKernels.matmulF32Q8_0MemSeg(
                        batchInput,
                        bData.segment,
                        bData.segmentByteOffset,
                        inputDim,
                        outputDim,
                        outBuffer,
                        batch * outputDim,
                    )
                }
            }
            else -> return null
        }
        return floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
    }

    /**
     * MemorySegment input × Q8 MemorySegment weights — zero-copy path that avoids
     * materializing input as a heap FloatArray.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> chooseQ8MemSegInputMatmul(
        aData: MemorySegmentBackedData,
        bData: Q8MemorySegmentMarker,
        batchSize: Int,
        inputDim: Int,
        outputDim: Int,
        a: Tensor<T, V>,
    ): Tensor<T, V> {
        val outBuffer = FloatArray(batchSize * outputDim)
        for (batch in 0 until batchSize) {
            val inputByteOffset = aData.segmentByteOffset + batch.toLong() * inputDim * 4
            JvmQuantizedVectorKernels.matmulF32Q8_0MemSegInput(
                aData.segment,
                inputByteOffset,
                bData.segment,
                bData.segmentByteOffset,
                inputDim,
                outputDim,
                outBuffer,
                batch * outputDim,
            )
        }
        return floatResult(Shape(batchSize, outputDim), a.dtype, outBuffer)
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        vectorFloatUnary(tensor, { vector ->
            val zero = FloatVector.zero(floatSpecies)
            vector.max(zero)
        }, { value ->
            if (value < 0f) 0f else value
        })?.let { return it }
        return super.relu(tensor)
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        val win = floatWinOf(tensor.data) ?: return super.silu(tensor)
        val buf = win.arr
        val off = win.off
        val n = tensor.shape.volume
        val out = FloatArray(n)
        for (i in 0 until n) {
            val x = buf[off + i]
            out[i] = x / (1f + kotlin.math.exp(-x))
        }
        return floatResult(Shape(tensor.shape.dimensions.copyOf()), tensor.dtype, out)
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            vectorFloatReduceAllSum<T, V>(tensor)?.let { return it }
        }
        return super.sum(tensor, dim)
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            val volume = tensor.shape.volume
            if (volume == 0) return super.mean(tensor, dim)
            vectorFloatReduceAllSum<T, V>(tensor)?.let { reduced ->
                val out = reduced as CpuTensor<T, V>
                @Suppress("UNCHECKED_CAST")
                val floatData = out.data as DenseFloatArrayTensorData<T>
                floatData.buffer[0] = floatData.buffer[0] / volume.toFloat()
                return reduced
            }
        }
        return super.mean(tensor, dim)
    }

    private fun <T : DType, V> vectorFloatBinary(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        vectorOp: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float
    ): Tensor<T, V>? {
        // Only FP32/FP16 supported for vector path
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null

        val aWin = floatWinOf(a.data) ?: return null
        val bWin = floatWinOf(b.data) ?: return null
        val aBuf = aWin.arr
        val aBase = aWin.off
        val bBuf = bWin.arr
        val bBase = bWin.off

        // Determine broadcasted output shape
        val outShape = try { broadcastShapes(a.shape, b.shape) } catch (e: IllegalArgumentException) { return null }
        val outVolume = outShape.volume
        val outBuffer = FloatArray(outVolume)

        val aVol = a.shape.volume
        val bVol = b.shape.volume

        // Case 1: exact shape match (fast path)
        if (a.shape == b.shape) {
            JvmVectorKernels.binaryFloat(aBuf, bBuf, outBuffer, outVolume, vectorOp, scalarOp, aOffset = aBase, bOffset = bBase)
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }

        // Case 2: scalar broadcast
        if (aVol == 1) {
            val aval = aBuf[aBase]
            val speciesLen = floatSpecies.length()
            var idx = 0
            val loopBound = floatSpecies.loopBound(outVolume)
            while (idx < loopBound) {
                val va = FloatVector.broadcast(floatSpecies, aval)
                val vb = FloatVector.fromArray(floatSpecies, bBuf, bBase + idx)
                vectorOp(va, vb).intoArray(outBuffer, idx)
                idx += speciesLen
            }
            while (idx < outVolume) {
                outBuffer[idx] = scalarOp(aval, bBuf[bBase + idx])
                idx++
            }
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }
        if (bVol == 1) {
            val bval = bBuf[bBase]
            val speciesLen = floatSpecies.length()
            var idx = 0
            val loopBound = floatSpecies.loopBound(outVolume)
            while (idx < loopBound) {
                val va = FloatVector.fromArray(floatSpecies, aBuf, aBase + idx)
                val vb = FloatVector.broadcast(floatSpecies, bval)
                vectorOp(va, vb).intoArray(outBuffer, idx)
                idx += speciesLen
            }
            while (idx < outVolume) {
                outBuffer[idx] = scalarOp(aBuf[aBase + idx], bval)
                idx++
            }
            return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
        }

        // Case 3: last-dimension broadcasting (bias add). Supports arbitrary leading dims.
        val aLast = if (a.shape.rank > 0) a.shape[a.shape.rank - 1] else 1
        val bLast = if (b.shape.rank > 0) b.shape[b.shape.rank - 1] else 1
        val outLast = outShape.dimensions.lastOrNull() ?: 1
        val groups = if (outLast == 0) 0 else outVolume / outLast
        if (groups > 0) {
            // b broadcasts across leading dims if its last dim == outLast and all other dims are 1 or equal
            val bIsBias = (b.shape.rank == 1 && bLast == outLast) || (
                b.shape.rank >= 1 && bLast == outLast && b.shape.dimensions.dropLast(1).all { it == 1 }
            )
            val aIsBias = (a.shape.rank == 1 && aLast == outLast) || (
                a.shape.rank >= 1 && aLast == outLast && a.shape.dimensions.dropLast(1).all { it == 1 }
            )
            if (bIsBias && aVol == outVolume) {
                val step = floatSpecies.length()
                val loopBoundTail = floatSpecies.loopBound(outLast)
                for (g in 0 until groups) {
                    val aOff = g * outLast
                    var idx = 0
                    while (idx < loopBoundTail) {
                        val va = FloatVector.fromArray(floatSpecies, aBuf, aBase + aOff + idx)
                        val vb = FloatVector.fromArray(floatSpecies, bBuf, bBase + idx)
                        vectorOp(va, vb).intoArray(outBuffer, aOff + idx)
                        idx += step
                    }
                    while (idx < outLast) {
                        outBuffer[aOff + idx] = scalarOp(aBuf[aBase + aOff + idx], bBuf[bBase + idx])
                        idx++
                    }
                }
                return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
            }
            if (aIsBias && bVol == outVolume) {
                val step = floatSpecies.length()
                val loopBoundTail = floatSpecies.loopBound(outLast)
                for (g in 0 until groups) {
                    val bOff = g * outLast
                    var idx = 0
                    while (idx < loopBoundTail) {
                        val va = FloatVector.fromArray(floatSpecies, aBuf, aBase + idx)
                        val vb = FloatVector.fromArray(floatSpecies, bBuf, bBase + bOff + idx)
                        vectorOp(va, vb).intoArray(outBuffer, bOff + idx)
                        idx += step
                    }
                    while (idx < outLast) {
                        outBuffer[bOff + idx] = scalarOp(aBuf[aBase + idx], bBuf[bBase + bOff + idx])
                        idx++
                    }
                }
                return floatResult(Shape(outShape.dimensions.copyOf()), a.dtype, outBuffer)
            }
        }

        // Fallback when complex broadcasting not supported here
        return null
    }

    private fun <T : DType, V> vectorFloatUnary(
        tensor: Tensor<T, V>,
        vectorOp: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float
    ): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val win = floatWinOf(tensor.data) ?: return null
        val volume = tensor.shape.volume
        val outBuffer = FloatArray(volume)
        JvmVectorKernels.unaryFloat(win.arr, outBuffer, volume, vectorOp, scalarOp, inputOffset = win.off)
        return floatResult(Shape(tensor.shape.dimensions.copyOf()), tensor.dtype, outBuffer)
    }

    /** Dense-FP32 window (array + base offset): plain array data at 0, slab-backed data (#1173) at its arrayOffset. */
    private class FloatWin(@JvmField val arr: FloatArray, @JvmField val off: Int)

    private fun floatWinOf(d: sk.ainet.lang.tensor.data.TensorData<*, *>): FloatWin? = when (d) {
        is FloatArrayTensorData<*> -> FloatWin(d.buffer, 0)
        is sk.ainet.lang.tensor.data.StorageFloatTensorData<*> -> {
            val st = d.storage
            st.checkAlive()
            FloatWin(st.floats!!, st.arrayOffset)
        }
        else -> null
    }

    private fun <T : DType> supportsFloatOps(a: Tensor<T, *>, b: Tensor<T, *>): Boolean {
        return supportsFloatOps(a) &&
            a.dtype == b.dtype &&
            a.shape == b.shape
    }

    private fun <T : DType> supportsFloatOps(tensor: Tensor<T, *>): Boolean {
        val dtype = tensor.dtype
        // BF16 was excluded here, which left BF16-tagged tensors matmul-only — every elementwise
        // and unary op fell through to the generic scalar path. All three float tags are backed by
        // FloatArray when produced by DenseTensorDataFactory, so they can use the vector kernels.
        // Callers re-check with `data as? FloatArrayTensorData`, so genuinely narrow-storage
        // tensors (Bf16/Fp16DenseTensorData) still fall through safely rather than being misread.
        return dtype == FP32::class || dtype == FP16::class || dtype == BF16::class
    }

    private fun <T : DType, V> chooseMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null
        if (a.shape.rank != 2 || b.shape.rank != 2) return null

        val aRows = a.shape[0]
        val aCols = a.shape[1]
        val bRows = b.shape[0]
        val bCols = b.shape[1]
        if (aCols != bRows) return null

        val m = aRows
        val n = bCols
        val k = aCols

        // ---- MemorySegment fast path ----
        val aMemSeg = a.data as? MemorySegmentBackedData
        val bMemSeg = b.data as? MemorySegmentBackedData
        if (aMemSeg != null && bMemSeg != null) {
            // Same fix as the transpose path above: use Arena.ofAuto so the
            // matmul output segment is GC-reclaimable. Per-call ofConfined()
            // leaks ~tens of MB per matmul, which over a 35-layer Gemma 4
            // forward pass exhausts the JVM direct-memory cap.
            val arena = Arena.ofAuto()
            val result = MemorySegmentTensorData<T>(Shape(m, n), arena)
            val blockedThresholdMS = 16 * 16
            if (m >= blockedThresholdMS || n >= blockedThresholdMS || k >= blockedThresholdMS) {
                JvmVectorKernels.matmulFloatBlockedMemSeg(
                    m, k, n,
                    aMemSeg.segment, aMemSeg.segmentByteOffset,
                    bMemSeg.segment, bMemSeg.segmentByteOffset,
                    result.segment, result.segmentByteOffset,
                    schedule = schedule,
                )
            } else {
                JvmVectorKernels.matmulFloatMemSeg(
                    m, k, n,
                    aMemSeg.segment, aMemSeg.segmentByteOffset,
                    bMemSeg.segment, bMemSeg.segmentByteOffset,
                    result.segment, result.segmentByteOffset,
                )
            }
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(result as TensorData<T, V>, this, a.dtype)
        }

        // ---- FloatArray path (plain arrays at offset 0, slab windows at their base — #1173) ----
        val aWin = floatWinOf(a.data) ?: return null
        val bWin = floatWinOf(b.data) ?: return null

        val work = m.toLong() * n.toLong() * k.toLong()
        val outBuffer = FloatArray(m * n)

        // Try BLAS for large sizes if enabled and available. The JNI shim takes whole arrays, so
        // it only serves offset-0 operands; slab windows go through the kernel SPI below, which
        // takes offsets natively.
        if (aWin.off == 0 && bWin.off == 0 && JvmCpuBackendConfig.blasEnabled && JvmBlas.isAvailable()) {
            val blasThreshold = 512L * 512L * 256L // tuneable
            if (work >= blasThreshold) {
                val ok = JvmBlas.sgemmRowMajorNN(m, n, k, 1f, aWin.arr, bWin.arr, outBuffer)
                if (ok) {
                    return floatResult(Shape(m, n), a.dtype, outBuffer)
                }
            }
        }

        // Small shapes are overhead-bound, not throughput-bound. The tile-blocked SPI kernel below
        // costs ~1ms per call whether it multiplies 1 row or 8 (Fp32GemvShapeBench: m=1 1.00ms,
        // m=8 1.08ms — 8x the arithmetic for 8% more time), so at decode sizes essentially all of
        // that is fixed cost. A decode step is m=1 by construction, and a model whose checkpoint
        // ships dense FP32 tensors does this per layer: Gemma 4 E2B has 70 such projections per
        // token and spent 73% of decode here. Do the small ones directly instead — contiguous in
        // `b` and `out`, one pass, no setup — and leave the tiled kernel the large shapes it wins.
        if (work <= SMALL_FP32_MATMUL_WORK) {
            val aBuf = aWin.arr
            val aBase = aWin.off
            val bBuf = bWin.arr
            val bBase = bWin.off
            for (i in 0 until m) {
                val aOff = aBase + i * k
                val outOff = i * n
                for (p in 0 until k) {
                    val av = aBuf[aOff + p]
                    if (av == 0f) continue
                    val bOff = bBase + p * n
                    for (j in 0 until n) {
                        outBuffer[outOff + j] += av * bBuf[bOff + j]
                    }
                }
            }
            return floatResult(Shape(m, n), a.dtype, outBuffer)
        }

        // Route through the kernel SPI — the registered provider
        // (Panama on JDK 21+, scalar otherwise) is tile-blocked and
        // handles small + large inputs in one path, so the previous
        // simple-vs-blocked fork is no longer needed.
        fp32MatmulKernel.matmul(
            aWin.arr, aWin.off, k,
            bWin.arr, bWin.off, n,
            outBuffer, 0, n,
            m, n, k,
        )
        return floatResult(Shape(m, n), a.dtype, outBuffer)
    }

    private fun <T : DType, V> vectorFloatReduceAllSum(tensor: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val win = floatWinOf(tensor.data) ?: return null
        val buffer = win.arr
        val base = win.off
        val n = tensor.shape.volume
        if (n == 0) return null
        // NOTE: For numerical reproducibility with Kotlin's FloatArray.sum(),
        // perform strict left-to-right scalar accumulation.
        var acc = 0.0f
        var idx = 0
        while (idx < n) {
            acc += buffer[base + idx]
            idx++
        }
        val outData = DenseFloatArrayTensorData<T>(Shape(), floatArrayOf(acc))
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, tensor.dtype)
    }
}
