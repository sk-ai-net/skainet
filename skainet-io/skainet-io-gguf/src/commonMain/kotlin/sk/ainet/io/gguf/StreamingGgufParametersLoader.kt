
package sk.ainet.io.gguf

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightByteOrder
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.io.openMappedFile
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.reflect.KClass

/**
 * Streaming GGUF parameters loader — the recommended path for loading GGUF models.
 *
 * Unlike [GgufParametersLoader] (which uses the legacy [GGUFReader] and rejects
 * quantized types), this loader:
 * - Uses [StreamingGGUFReader] for memory-efficient parsing
 * - Supports quantized types ([SUPPORTED_TENSOR_TYPES]) as packed [TensorData]
 * - Loads tensor data on-demand without heap-loading the full file
 * - Preserves quantized layout through the loading pipeline
 *
 * For F32 and I32 tensors, data is returned as standard dense arrays.
 * For quantized tensors, data is returned as packed block storage
 * (e.g., [Q4_KBlockTensorData], [Q8_0BlockTensorData]).
 *
 * A file containing tensors outside [SUPPORTED_TENSOR_TYPES] (e.g. Q4_1)
 * fails fast: [load] throws before any tensor is delivered, naming the
 * offending tensors and the supported set, instead of silently skipping
 * them and letting the missing weights crash the forward pass later
 * (#919).
 */
public class StreamingGgufParametersLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    /**
     * Keep `F16` source tensors in their on-disk 2-bytes-per-element layout instead of widening
     * them to FP32 at load. Off by default — flip via `withPolicy(Require(FP16))`.
     */
    private val keepF16Native: Boolean = false,
    /**
     * Keep `BF16` source tensors packed. Off by default — flip via `withPolicy(Require(BF16))`.
     */
    private val keepBf16Native: Boolean = false,
    /**
     * The form weights should take in memory — encoding × byte order × shape × residency as one
     * decision (#1109, #1115, #1159).
     *
     * `null` (the default) means [WeightForm.AS_STORED_ON_HEAP]: the file's bytes, its order, on
     * the heap — the loader's historical behaviour. Callers who know their device pass a form;
     * callers who don't let `WeightFormResolver`/[ResolvedGguf] resolve one from what the file
     * holds, what the device is, and what the backend's kernels can feed.
     */
    private val weightForm: WeightForm? = null,
    /**
     * Per-tensor forms — the *user wins* channel (#1144).
     *
     * Precedence, explicit and documented: **this function > [weightForm] > the as-stored-on-heap
     * default**. Whatever you return for a tensor outranks every resolver and every profile —
     * including the deliberately blunt `WeightForm(DequantizeTo(FP32), residency = HEAP)`
     * ("everything dense, on the managed heap"). Return `null` for tensors you have no opinion on;
     * they fall through to [weightForm].
     *
     * The intended producer is `WeightFormResolver`/`resolveWeightForms` via [ResolvedGguf], which
     * resolves per tensor from the file × profile × kernel capability — but the contract is the
     * same for a hand-written lambda: the loader carries and obeys, it does not decide.
     */
    private val weightFormFor: ((tensorName: String) -> WeightForm?)? = null,
    /**
     * Where conversions are reported (#1117).
     *
     * A weight that arrives in the form it will be used in costs nothing and says nothing. One that
     * is re-encoded on the way in — dequantized because no kernel can feed its encoding, widened
     * because it is a narrow float, or widened because ternary packed storage does not exist yet
     * (#1033) — emits a `TraceEvent.AdapterInserted` naming the tensor and the sizes either side.
     * That is the difference between "why is this model 3 GB" being answerable and being folklore.
     *
     * Defaults to [NoopTraceSink]: nothing is recorded and nothing is allocated.
     */
    private val traceSink: TraceSink = NoopTraceSink,
    /**
     * The bit order I2_S (type 36) payloads in this file are in — a property of the converter
     * that wrote the file, not recoverable from the bytes (#1140, see [I2sGgufLayout]). Defaults
     * to [I2sGgufLayout.GROUP_128], the BitNet.cpp x86 pipeline behind the commonly published
     * GGUFs. Wrong-layout loads fail fast on code 3 where possible, but a misdeclared layout can
     * also decode silently wrong — this knob is the caller's responsibility.
     */
    private val i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
) : ParametersLoader {

    /**
     * Report that [tensorName] was re-encoded from [from] to [to] on the way in.
     *
     * Guarded on [TraceSink.isEnabled] so the common case builds no `Format`s and allocates
     * nothing. The tensor is named by parsing its GGUF name as a [TensorId] — `blk.0.attn_q.weight`
     * is already dotted, and renders back as itself.
     */
    private fun traceConversion(
        kind: String,
        tensorName: String,
        from: Format,
        to: Format,
        bytesBefore: Long,
        bytesAfter: Long,
    ) {
        if (!traceSink.isEnabled) return
        traceSink.emit(
            TraceEvent.AdapterInserted(
                kind = kind,
                from = from,
                to = to,
                bytes = bytesAfter,
                target = runCatching { TensorId.parse(tensorName) }.getOrNull(),
                scope = ScopeKind.MODEL,
                bytesBefore = bytesBefore,
            ),
        )
    }

    /** The dense FP32 size of [elements] — what every widening in this loader converts to. */
    private fun denseFp32Bytes(elements: Long): Long = elements * 4

    /** Whether [form] asks for the one requantization this loader has (#1150). */
    private fun planesRequested(form: WeightForm): Boolean =
        (form.encoding as? EncodingRequest.RequantizeTo)?.encoding ==
            sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_PLANES

    /**
     * Requantize [values] (row-major `[out, in]` — validateForm pinned the orientation) into a
     * packed [sk.ainet.lang.tensor.data.BitNetPlanesTensorData] (#1150): 8 trit planes + FP16
     * per-row scales, the lm_head format the planes kernel pack serves. Traced — a requantize is
     * a conversion someone should be able to see.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> planesTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape,
        tensorName: String,
        values: FloatArray,
        bytesBefore: Long,
        fromFormat: Format,
    ): Tensor<T, V> {
        require(shape.rank == 2) {
            "tensor '$tensorName': RequantizeTo(BITNET_PLANES) needs a 2-D weight, got rank ${shape.rank}"
        }
        require(dtype == FP32::class) {
            "tensor '$tensorName': BITNET_PLANES weights are logically FP32; requested $dtype"
        }
        val data = sk.ainet.lang.tensor.data.BitNetPlanesTensorData.fromFloats(shape, values)
        traceConversion(
            kind = "requantize-planes",
            tensorName = tensorName,
            from = fromFormat,
            to = Format(FP32, sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_PLANES),
            bytesBefore = bytesBefore,
            bytesAfter = data.packedData.size.toLong(),
        )
        return ctx.fromData(data as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
    }

    /** The uniform form: [weightForm] if given, else the historical as-stored-on-heap default. */
    private val form: WeightForm = weightForm ?: WeightForm.AS_STORED_ON_HEAP

    /**
     * The form [tensorName] loads under — the precedence order of [weightFormFor], validated the
     * same way the uniform [form] was at construction.
     */
    private fun formFor(tensorName: String): WeightForm {
        val perTensor = weightFormFor?.invoke(tensorName) ?: return form
        validateForm(perTensor, "weightFormFor('$tensorName')")
        return perTensor
    }

    init {
        validateForm(form, "weightForm")
    }

    private fun validateForm(form: WeightForm, where: String) {
        require(form.order == WeightByteOrder.AS_STORED || form.shape == WeightShapeOrientation.OUT_IN) {
            "$where: WeightByteOrder.KERNEL_FEED needs WeightShapeOrientation.OUT_IN: feed order is " +
                "defined relative to a [out, in] weight — which block is 'block b of output row o' " +
                "has no answer while the tensor is still labelled in the file's `ne` order."
        }
        val requested = form.encoding
        if (requested is EncodingRequest.RequantizeTo) {
            // #1150: the one requantizer this loader has. Everything else still fails eagerly.
            require(requested.encoding == sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_PLANES) {
                "$where: EncodingRequest.RequantizeTo(${requested.encoding.name}) is not supported " +
                    "by this loader: re-quantizing a weight the file does not already carry needs a " +
                    "quantizer per target encoding, and only BITNET_PLANES exists here (#1150)."
            }
            require(form.shape == WeightShapeOrientation.OUT_IN) {
                "$where: RequantizeTo(BITNET_PLANES) needs WeightShapeOrientation.OUT_IN — the " +
                    "format's scales are per output row, which has no answer while the tensor is " +
                    "still labelled in the file's `ne` order."
            }
        }
        val dequantTarget = (requested as? EncodingRequest.DequantizeTo)?.dtype
        require(dequantTarget == null || dequantTarget == FP32) {
            "$where: EncodingRequest.DequantizeTo(${dequantTarget?.name}) is not supported: this " +
                "loader dequantizes to FP32 only."
        }
    }

    /**
     * The shape this loader reports for [tensorInfo], honouring [weightOrientation]: a 2-D weight
     * is reversed for `OUT_IN`, everything else is passed through as the file has it. Only 2-D
     * tensors are touched — a 1-D bias or norm has no orientation to get wrong.
     */
    private fun shapeOf(tensorInfo: StreamingTensorInfo, form: WeightForm): Shape {
        val dims = tensorInfo.shape.map { it.toInt() }
        val reverses = form.shape == WeightShapeOrientation.OUT_IN
        val ordered = if (reverses && dims.size == 2) dims.reversed() else dims
        return Shape(*ordered.toIntArray())
    }

    @Suppress("UNCHECKED_CAST")
    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        val source = sourceProvider()
        // MAPPED staging needs a file to map; a Blob or an in-memory source has no path and
        // silently stays on the heap, which is the documented fallback rather than a failure.
        // With per-tensor forms the file is mapped whenever any tensor *might* ask for it.
        val mightMap = form.residency == WeightResidency.MAPPED || weightFormFor != null
        val mapped = if (mightMap) source.filePath?.let { openMappedFile(it) } else null
        try {
        StreamingGGUFReader.open(source).use { reader ->
            val tensors = reader.tensors
            failFastOnUnsupportedTensorTypes(tensors)
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                // NeoGPU's I2_S converter emits a companion `<name>_scale` F32 scalar per ternary
                // weight; it is consumed by the I2_S branch below (folded into the BITNET_B1_58
                // trailer), never delivered as a parameter of its own.
                if (isI2sCompanionScale(tensorInfo, tensors)) {
                    current += 1
                    onProgress(current, total, tensorInfo.name)
                    continue
                }
                val tensorForm = formFor(tensorInfo.name)
                val shape = shapeOf(tensorInfo, tensorForm)
                // A dense F32 tensor under MAPPED staging never reaches the heap: it is a view over
                // file-backed pages. Everything else reads its bytes (out of the mapping when there
                // is one — one page-cache copy instead of a channel read).
                val mappedFloats: Tensor<T, V>? =
                    if (mapped != null && tensorForm.residency == WeightResidency.MAPPED &&
                        tensorInfo.tensorType == GGMLQuantizationType.F32 && dtype == FP32::class
                    ) {
                        @Suppress("UNCHECKED_CAST")
                        ctx.fromData(
                            mapped.denseFloats<T>(tensorInfo.absoluteDataOffset, shape) as sk.ainet.lang.tensor.data.TensorData<T, V>,
                            dtype,
                        )
                    } else {
                        null
                    }
                if (mappedFloats != null) {
                    onTensorLoaded(tensorInfo.name, mappedFloats)
                    current += 1
                    onProgress(current, total, tensorInfo.name)
                    continue
                }
                // #1189/#1192: a GGML block-format tensor under MAPPED staging is the packed counterpart of the
                // dense case above — served as a view over the file-backed pages, blocks left in
                // canonical row-major order, never a heap ByteArray. Only when nothing asks for a
                // different form: a dequantize/planes request or KERNEL_FEED order needs the bytes
                // (or values) materialized anyway, and a platform without off-heap packed storage
                // returns null and falls through to heap staging as before.
                val mappedPacked: Tensor<T, V>? =
                    if (mapped != null && tensorForm.residency == WeightResidency.MAPPED &&
                        dtype == FP32::class &&
                        tensorForm.order != WeightByteOrder.KERNEL_FEED &&
                        tensorForm.encoding !is EncodingRequest.DequantizeTo &&
                        !planesRequested(tensorForm)
                    ) {
                        val enc = when (tensorInfo.tensorType) {
                            GGMLQuantizationType.Q4_K -> sk.ainet.lang.tensor.storage.TensorEncoding.Q4_K
                            GGMLQuantizationType.Q6_K -> sk.ainet.lang.tensor.storage.TensorEncoding.Q6_K
                            GGMLQuantizationType.Q5_K -> sk.ainet.lang.tensor.storage.TensorEncoding.Q5_K
                            GGMLQuantizationType.Q8_0 -> sk.ainet.lang.tensor.storage.TensorEncoding.Q8_0
                            GGMLQuantizationType.Q4_0 -> sk.ainet.lang.tensor.storage.TensorEncoding.Q4_0
                            GGMLQuantizationType.Q5_0 -> sk.ainet.lang.tensor.storage.TensorEncoding.Q5_0
                            GGMLQuantizationType.Q5_1 -> sk.ainet.lang.tensor.storage.TensorEncoding.Q5_1
                            // #1203: only when the payload is already SEQUENTIAL (no permutation
                            // needed) and the trailing bytes are genuinely the scale (no companion
                            // tensor overrides them) -- otherwise fall through to i2sTensor's
                            // repack, unchanged.
                            GGMLQuantizationType.I2_S ->
                                if (i2sTrailerScaleIsMappable(tensorInfo, tensors, source, i2sLayout)) {
                                    sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_B1_58
                                } else {
                                    null
                                }
                            else -> null
                        }
                        enc?.let { encoding ->
                            mapped.packedTensor(tensorInfo.absoluteDataOffset, shape, encoding)?.let {
                                @Suppress("UNCHECKED_CAST")
                                ctx.fromData(it as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                            }
                        }
                    } else {
                        null
                    }
                if (mappedPacked != null) {
                    onTensorLoaded(tensorInfo.name, mappedPacked)
                    current += 1
                    onProgress(current, total, tensorInfo.name)
                    continue
                }
                val rawBytes = mapped?.bytes(tensorInfo.absoluteDataOffset, tensorInfo.nBytes.toInt())
                    ?: reader.loadTensorData(tensorInfo)

                val tensor: Tensor<T, V>? = when (tensorInfo.tensorType) {
                    GGMLQuantizationType.F32 -> {
                        when (dtype) {
                            // The freshly decoded array is loader-owned — wrap it zero-copy
                            // instead of paying the factory's defensive copy (#782).
                            FP32::class -> if (planesRequested(tensorForm)) {
                                planesTensor(
                                    ctx, dtype, shape, tensorInfo.name, bytesToFloatArray(rawBytes),
                                    rawBytes.size.toLong(), Format.dense(FP32),
                                )
                            } else {
                                ctx.wrapFloatArray<T, Float>(shape, dtype, bytesToFloatArray(rawBytes)) as Tensor<T, V>
                            }
                            else -> null
                        }
                    }

                    GGMLQuantizationType.I32 -> {
                        val ints = bytesToIntArray(rawBytes)
                        when (dtype) {
                            Int32::class -> ctx.fromIntArray<T, Int>(shape, dtype, ints) as Tensor<T, V>
                            else -> null
                        }
                    }

                    GGMLQuantizationType.F16 -> when (dtype) {
                        FP32::class -> if (keepF16Native) {
                            // Zero-widening path: hand the on-disk bytes straight through as
                            // packed binary16. Consumers still see Float on read.
                            @Suppress("UNCHECKED_CAST")
                            val packed = Fp16DenseTensorData(shape, rawBytes)
                            ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                        } else {
                            // Loader-owned widened array — zero-copy wrap (#782).
                            if (planesRequested(tensorForm)) {
                                planesTensor(
                                    ctx, dtype, shape, tensorInfo.name, dequantF16(rawBytes),
                                    rawBytes.size.toLong(), Format.dense(FP16),
                                )
                            } else {
                                traceConversion(
                                    "widen-f16", tensorInfo.name, Format.dense(FP16), Format.dense(FP32),
                                    rawBytes.size.toLong(), denseFp32Bytes(tensorInfo.nElements),
                                )
                                ctx.wrapFloatArray<T, Float>(shape, dtype, dequantF16(rawBytes)) as Tensor<T, V>
                            }
                        }
                        else -> null
                    }

                    GGMLQuantizationType.BF16 -> when (dtype) {
                        FP32::class -> if (keepBf16Native) {
                            @Suppress("UNCHECKED_CAST")
                            val packed = Bf16DenseTensorData(shape, rawBytes)
                            ctx.fromData<T, V>(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
                        } else {
                            // Loader-owned widened array — zero-copy wrap (#782).
                            traceConversion(
                                "widen-bf16", tensorInfo.name, Format.dense(BF16), Format.dense(FP32),
                                rawBytes.size.toLong(), denseFp32Bytes(tensorInfo.nElements),
                            )
                            ctx.wrapFloatArray<T, Float>(shape, dtype, dequantBF16(rawBytes)) as Tensor<T, V>
                        }
                        else -> null
                    }

                    GGMLQuantizationType.Q4_K,
                    GGMLQuantizationType.Q5_K,
                    GGMLQuantizationType.Q6_K,
                    GGMLQuantizationType.Q8_0,
                    GGMLQuantizationType.Q4_0,
                    GGMLQuantizationType.Q5_0,
                    GGMLQuantizationType.Q5_1,
                    GGMLQuantizationType.TQ1_0,
                    GGMLQuantizationType.TQ2_0 -> quantizedTensor(ctx, dtype, shape, tensorInfo, rawBytes, tensorForm)

                    GGMLQuantizationType.I2_S -> i2sTensor(
                        ctx, dtype, shape, tensorInfo, rawBytes, tensorForm,
                        scale = resolveI2sScale(tensorInfo, tensors, reader, source, i2sLayout),
                    )

                    else -> throw IllegalStateException(
                        "StreamingGgufParametersLoader: tensor '${tensorInfo.name}' of type " +
                            "${tensorInfo.tensorType} passed the load-time pre-scan but has no load " +
                            "branch — SUPPORTED_TENSOR_TYPES and this when-expression have drifted. " +
                            "Please report this as a bug."
                    )
                }

                if (tensor != null) {
                    onTensorLoaded(tensorInfo.name, tensor)
                }

                current += 1
                onProgress(current, total, tensorInfo.name)
            }
        }
        } finally {
            mapped?.close()
        }
    }

    /**
     * Materialize a quantized tensor according to [quantPolicy].
     *
     * DEQUANTIZE_TO_FP32 (#782): the packed bytes are unpacked block-by-block
     * straight into one destination `FloatArray` (the shared [DequantOps]
     * kernels write each block into the single output array — no boxed values,
     * no per-tensor intermediate), and the destination is wrapped zero-copy.
     * Peak transient allocation per tensor is the packed source bytes.
     *
     * Any other policy (or a non-float destination dtype) preserves the packed
     * block storage exactly as before.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> quantizedTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape,
        tensorInfo: StreamingTensorInfo,
        rawBytes: ByteArray,
        tensorForm: WeightForm,
    ): Tensor<T, V> {
        if (planesRequested(tensorForm) && dtype == FP32::class) {
            val dest = DequantOps.dequantFromBytes(rawBytes, tensorInfo.tensorType, tensorInfo.nElements.toInt())
            return planesTensor(
                ctx, dtype, shape, tensorInfo.name, dest,
                rawBytes.size.toLong(), ggufFormat(tensorInfo.tensorType, rawBytes.size.toLong()),
            )
        }
        if (tensorForm.encoding is EncodingRequest.DequantizeTo &&
            (dtype == FP32::class || dtype == FP16::class)
        ) {
            val dest = DequantOps.dequantFromBytes(rawBytes, tensorInfo.tensorType, tensorInfo.nElements.toInt())
            traceConversion(
                kind = "dequantize-on-load",
                tensorName = tensorInfo.name,
                from = ggufFormat(tensorInfo.tensorType, rawBytes.size.toLong()),
                to = Format.dense(FP32),
                bytesBefore = rawBytes.size.toLong(),
                bytesAfter = denseFp32Bytes(tensorInfo.nElements),
            )
            return ctx.wrapFloatArray<T, Float>(shape, dtype, dest) as Tensor<T, V>
        }
        if (tensorInfo.tensorType == GGMLQuantizationType.TQ1_0 || tensorInfo.tensorType == GGMLQuantizationType.TQ2_0) {
            // #1033: ternary tensors decode correctly, but there is no packed ternary `TensorData`
            // with per-block scales yet — it arrives with the ternary kernels (#1040/#1041). Until
            // then every policy widens them to FP32: right values, no memory saving.
            require(dtype == FP32::class || dtype == FP16::class) {
                "tensor '${tensorInfo.name}' is ${tensorInfo.tensorType}; ternary tensors currently " +
                    "load as FP32, so the requested dtype $dtype is not supported"
            }
            val dest = DequantOps.dequantFromBytes(rawBytes, tensorInfo.tensorType, tensorInfo.nElements.toInt())
            // Worth seeing precisely because no policy asked for it: a 1.6-bit weight arriving as
            // FP32 is a ~20× widening that no flag on this loader can currently turn off (#1033).
            traceConversion(
                kind = "widen-ternary-no-packed-storage",
                tensorName = tensorInfo.name,
                from = ggufFormat(tensorInfo.tensorType, rawBytes.size.toLong()),
                to = Format.dense(FP32),
                bytesBefore = rawBytes.size.toLong(),
                bytesAfter = denseFp32Bytes(tensorInfo.nElements),
            )
            @Suppress("UNCHECKED_CAST")
            return ctx.wrapFloatArray<T, Float>(shape, dtype, dest) as Tensor<T, V>
        }
        val packed = when (tensorInfo.tensorType) {
            GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_K -> Q5_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q6_K -> Q6_KBlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q8_0 -> Q8_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q4_0 -> Q4_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_0 -> Q5_0BlockTensorData.fromRawBytes(shape, rawBytes)
            GGMLQuantizationType.Q5_1 -> Q5_1BlockTensorData.fromRawBytes(shape, rawBytes)
            else -> throw IllegalStateException(
                "quantizedTensor called for non-quantized type ${tensorInfo.tensorType}"
            )
        }
        val delivered = if (tensorForm.order == WeightByteOrder.KERNEL_FEED) feedOrdered(packed, tensorInfo) else packed
        return ctx.fromData(delivered as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
    }

    /**
     * Whether [tensorInfo] is a NeoGPU-converter companion scale — an F32 scalar named
     * `<weight>_scale` next to an I2_S tensor of that name. Consumed by [resolveI2sScale],
     * skipped as a parameter.
     */
    private fun isI2sCompanionScale(
        tensorInfo: StreamingTensorInfo,
        tensors: List<StreamingTensorInfo>,
    ): Boolean =
        tensorInfo.tensorType == GGMLQuantizationType.F32 &&
            tensorInfo.name.endsWith("_scale") &&
            tensors.any {
                it.tensorType == GGMLQuantizationType.I2_S &&
                    "${it.name}_scale" == tensorInfo.name
            }

    /**
     * Materialize an I2_S tensor (#1140): repack the payload into the sequential `BITNET_B1_58`
     * order under [i2sLayout] (code 3 fails fast in [I2sRepack]), fold [scale] into the trailer,
     * and keep it packed — 0.25 bytes per weight instead of the #1033 FP32 widening. A
     * `DequantizeTo` form still gets dense FP32, decoded through the same codec the kernels are
     * defined against.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> i2sTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        shape: Shape,
        tensorInfo: StreamingTensorInfo,
        rawBytes: ByteArray,
        tensorForm: WeightForm,
        scale: Float,
    ): Tensor<T, V> {
        require(dtype == FP32::class || dtype == FP16::class) {
            "tensor '${tensorInfo.name}' is I2_S; ternary weights are logically FP32, so the " +
                "requested dtype $dtype is not supported"
        }
        val payload = I2sRepack.toSequentialPayload(rawBytes, tensorInfo.nElements.toInt(), i2sLayout)
        val packedBytes = I2sRepack.withScale(payload, scale)
        traceConversion(
            kind = "repack-i2s",
            tensorName = tensorInfo.name,
            from = ggufFormat(GGMLQuantizationType.I2_S, rawBytes.size.toLong()),
            to = Format(FP32, sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_B1_58),
            bytesBefore = rawBytes.size.toLong(),
            bytesAfter = packedBytes.size.toLong(),
        )
        if (planesRequested(tensorForm)) {
            return planesTensor(
                ctx, dtype, shape, tensorInfo.name,
                sk.ainet.lang.memory.TernaryCodec.decodeBitNet(packedBytes, tensorInfo.nElements.toInt()),
                rawBytes.size.toLong(),
                Format(FP32, sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_B1_58),
            )
        }
        if (tensorForm.encoding is EncodingRequest.DequantizeTo) {
            val dest = sk.ainet.lang.memory.TernaryCodec.decodeBitNet(packedBytes, tensorInfo.nElements.toInt())
            traceConversion(
                kind = "widen-i2s",
                tensorName = tensorInfo.name,
                from = Format(FP32, sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_B1_58),
                to = Format.dense(FP32),
                bytesBefore = packedBytes.size.toLong(),
                bytesAfter = denseFp32Bytes(tensorInfo.nElements),
            )
            return ctx.wrapFloatArray<T, Float>(shape, dtype, dest) as Tensor<T, V>
        }
        val packed = i2sPackedTensorData(shape, packedBytes)
        return ctx.fromData(packed as sk.ainet.lang.tensor.data.TensorData<T, V>, dtype)
    }

    /**
     * [packedBytes] as a [sk.ainet.lang.tensor.data.BitNetB158TensorData] — off the ART heap
     * (#1202) when the payload is at least [sk.ainet.lang.memory.plan.PlannerProfile.OFF_HEAP_THRESHOLD]
     * and the platform supports it; otherwise the historical heap-backed constructor.
     *
     * A 2B-parameter BitNet model's ternary weights are ~0.25 bytes/weight — roughly 500MB
     * resident as plain `ByteArray`s, which alone can exceed Android's default ART heap cap. This
     * is independent of *residency* (`WeightResidency.MAPPED` isn't reachable for a repacked
     * payload — the bytes are no longer the file's own — see `AllocationResolver`): it's purely
     * about not holding the repack's result in managed-heap memory once it's large enough to
     * matter.
     */
    private fun i2sPackedTensorData(shape: Shape, packedBytes: ByteArray): sk.ainet.lang.tensor.data.BitNetB158TensorData {
        val offHeapThreshold = sk.ainet.lang.memory.plan.PlannerProfile.OFF_HEAP_THRESHOLD
        if (packedBytes.size < offHeapThreshold || !sk.ainet.lang.memory.PlatformStorage.supports(sk.ainet.lang.tensor.storage.MemoryDomain.HOST_OFFHEAP)) {
            return sk.ainet.lang.tensor.data.BitNetB158TensorData(shape, packedBytes)
        }
        val storage = sk.ainet.lang.memory.PlatformStorage.allocate(
            bytes = packedBytes.size.toLong(),
            domain = sk.ainet.lang.tensor.storage.MemoryDomain.HOST_OFFHEAP,
            scope = sk.ainet.lang.memory.ScopeKind.MODEL,
            sink = traceSink,
        )
        storage.copyFrom(packedBytes)
        return sk.ainet.lang.tensor.data.BitNetB158TensorData.fromStorage(shape, storage)
    }

    /**
     * [packed] with its blocks permuted into the order the packed matmul kernels read, and *saying
     * so* (#1120).
     *
     * The permutation is `TensorView.prepack`, which already owns it and emits the conversion on
     * the trace (#1117). What #1120 adds is the second half: the result is rebuilt as a
     * `TensorData` carrying `BlockOrder.INPUT_BLOCK_MAJOR`, so every reader is right about the same
     * bytes — the kernels address them in feed order deliberately, and anything decoding through
     * the view or `toFloatArray()` walks the logical grid and fetches each block from where this
     * order put it. Before that, feed-order bytes in a type claiming to be canonical decoded to
     * plausible garbage (#1124, #973, #968).
     */
    private fun feedOrdered(
        packed: sk.ainet.lang.tensor.storage.PackedBlockStorage,
        tensorInfo: StreamingTensorInfo,
    ): sk.ainet.lang.tensor.storage.PackedBlockStorage {
        val shape = packed.shape
        if (shape.rank != 2 || shape[1] % packed.blockSize != 0) return packed
        val prepacked = packed.packedView.prepack(sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR, sink = traceSink)
        val bytes = (prepacked.storage as? sk.ainet.lang.memory.Storage.Heap)?.bytes ?: return packed
        val order = sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR
        return when (tensorInfo.tensorType) {
            GGMLQuantizationType.Q4_K -> Q4_KBlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q5_K -> Q5_KBlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q6_K -> Q6_KBlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q8_0 -> Q8_0BlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q4_0 -> Q4_0BlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q5_0 -> Q5_0BlockTensorData(shape, bytes, order)
            GGMLQuantizationType.Q5_1 -> Q5_1BlockTensorData(shape, bytes, order)
            else -> packed
        }
    }

    private fun bytesToFloatArray(bytes: ByteArray): FloatArray {
        val count = bytes.size / 4
        return FloatArray(count) { i ->
            val off = i * 4
            Float.fromBits(
                (bytes[off].toInt() and 0xFF) or
                    ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[off + 3].toInt() and 0xFF) shl 24)
            )
        }
    }

    private fun bytesToIntArray(bytes: ByteArray): IntArray {
        val count = bytes.size / 4
        return IntArray(count) { i ->
            val off = i * 4
            (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8) or
                ((bytes[off + 2].toInt() and 0xFF) shl 16) or
                ((bytes[off + 3].toInt() and 0xFF) shl 24)
        }
    }

    private fun dequantF16(bytes: ByteArray): FloatArray {
        val count = bytes.size / 2
        return FloatArray(count) { i ->
            val off = i * 2
            val halfBits = (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8)
            halfToFloat(halfBits)
        }
    }

    private fun dequantBF16(bytes: ByteArray): FloatArray {
        val count = bytes.size / 2
        return FloatArray(count) { i ->
            val off = i * 2
            val bf16Bits = (bytes[off].toInt() and 0xFF) or
                ((bytes[off + 1].toInt() and 0xFF) shl 8)
            Float.fromBits(bf16Bits shl 16)
        }
    }

    public companion object {

        /**
         * The tensor types [load] can materialize. The when-expression in [load]
         * and the eager pre-scan both derive from this set, so a type added to
         * one place cannot silently drift from the other.
         */
        public val SUPPORTED_TENSOR_TYPES: Set<GGMLQuantizationType> = setOf(
            GGMLQuantizationType.F32,
            GGMLQuantizationType.I32,
            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16,
            GGMLQuantizationType.Q4_0,
            GGMLQuantizationType.Q5_0,
            GGMLQuantizationType.Q5_1,
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_0,
            // #1033: ternary. Decoded through the reference codec next to their
            // TensorEncoding descriptor, so the loader and the ternary kernels
            // cannot read the same bytes differently.
            GGMLQuantizationType.TQ1_0,
            GGMLQuantizationType.TQ2_0,
            // #1140: BitNet.cpp / NeoGPU ternary. Repacked at load into the
            // sequential BITNET_B1_58 layout and kept packed (0.25 B/weight)
            // — the first ternary type that does NOT widen to FP32 (#1033).
            GGMLQuantizationType.I2_S,
        )

        private const val MAX_LISTED_TENSORS = 8

        /**
         * Eager pre-scan over the file's tensor directory: throws before any
         * tensor is delivered if the file contains types this loader cannot
         * materialize. This follows the RFC's "fail before execution" rule
         * (see [withPolicy]) — the alternative, skipping the tensor, produces
         * a model with silently missing weights whose failure surfaces far
         * away in the forward pass (#919).
         */
        internal fun failFastOnUnsupportedTensorTypes(tensors: List<StreamingTensorInfo>) {
            val unsupported = tensors.filter { it.tensorType !in SUPPORTED_TENSOR_TYPES }
            if (unsupported.isEmpty()) return

            val listed = unsupported.take(MAX_LISTED_TENSORS).joinToString(", ") {
                val type = if (it.isUnknownType) "unknown type value ${it.rawTypeValue}" else it.tensorType.name
                "'${it.name}' ($type)"
            }
            val more = if (unsupported.size > MAX_LISTED_TENSORS) {
                " and ${unsupported.size - MAX_LISTED_TENSORS} more"
            } else {
                ""
            }
            throw IllegalArgumentException(
                "GGUF contains ${unsupported.size} tensor(s) with quantization types this loader " +
                    "does not support: $listed$more. Supported types: " +
                    "${SUPPORTED_TENSOR_TYPES.joinToString(", ") { it.name }}. " +
                    "Re-quantize the model to a supported format (e.g. Q8_0, Q4_0, Q4_K or F16).",
            )
        }

        /**
         * Convenience constructor that takes a [DTypePolicy] and
         * validates it against the dtypes the GGUF loader supports
         * today. The validator runs eagerly — if the requested
         * policy can never be satisfied by this loader (e.g.
         * `Require(Int8)` against a GGUF file: this loader doesn't
         * cast), an [IllegalArgumentException] is raised before the
         * loader is constructed, exactly matching the RFC's
         * "fail before execution" rule.
         *
         * Current per-source behaviour the validator enforces:
         * - GGUF `F32` / `I32` / `Q4_K` / `Q8_0` are always
         *   preserved verbatim — any policy that admits the
         *   matching dtype passes.
         * - GGUF `F16` / `BF16` always dequant to FP32 in this
         *   loader today (no KEEP_NATIVE GGUF path yet). A policy
         *   of `Require(BF16)` or `Require(FP16)` therefore fails
         *   eagerly; use `Any`, `Prefer`, or `OneOf` containing
         *   `FP32` if you want the adaptive dequant behaviour.
         *
         * The validator is conservative — it doesn't open the GGUF
         * file to check which dtypes are actually present. A
         * policy that's satisfiable in principle but happens to
         * conflict with the specific file's tensors will surface at
         * iteration time via the `null`-return path in [load].
         * Tensor *types* outside [SUPPORTED_TENSOR_TYPES], by
         * contrast, fail eagerly once the file is opened — see
         * [failFastOnUnsupportedTensorTypes].
         */
        public fun withPolicy(
            sourceProvider: () -> RandomAccessSource,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        ): StreamingGgufParametersLoader {
            validatePolicy(policy)
            return StreamingGgufParametersLoader(
                sourceProvider = sourceProvider,
                onProgress = onProgress,
                keepF16Native = keepsNative(policy, FP16),
                keepBf16Native = keepsNative(policy, BF16),
            )
        }

        /**
         * Whether [policy] asks for [native] tensors to stay in their on-disk 16-bit layout.
         *
         * Only the format the policy actually names is kept — neither narrow format can be turned
         * into the other without a lossy re-encode, so `Require(BF16)` must still widen F16 sources.
         */
        internal fun keepsNative(policy: DTypePolicy, native: DType): Boolean = when (policy) {
            DTypePolicy.Any -> false
            is DTypePolicy.Require -> policy.target == native
            is DTypePolicy.Prefer -> policy.target == native
            is DTypePolicy.OneOf -> native in policy.allowed
        }

        internal fun validatePolicy(policy: DTypePolicy) {
            when (policy) {
                DTypePolicy.Any -> Unit
                is DTypePolicy.Prefer -> Unit
                is DTypePolicy.OneOf -> Unit
                is DTypePolicy.Require -> when (policy.target) {
                    // FP16 / BF16 are satisfiable for sources already in that format: the loader
                    // hands the packed bytes through via Fp16/Bf16DenseTensorData. Sources in any
                    // other format still widen to FP32 rather than being re-encoded.
                    FP32, FP16, BF16 -> Unit
                    else -> throw IllegalArgumentException(
                        "StreamingGgufParametersLoader: Require(${policy.target.name}) is not satisfiable — " +
                            "this loader preserves source tensors (dense FP32/Int32 or packed quantized " +
                            "blocks) and does not cast between dtypes. Use Any to inherit the source dtype, " +
                            "or open a follow-up to add a ${policy.target.name} cast path.",
                    )
                }
            }
        }
    }

    private fun halfToFloat(hbits: Int): Float {
        val sign = (hbits and 0x8000) shl 16
        val exp = (hbits and 0x7C00) shr 10
        val mant = hbits and 0x03FF

        return when (exp) {
            0 -> {
                if (mant == 0) Float.fromBits(sign)
                else {
                    var m = mant; var e = -14
                    while ((m and 0x400) == 0) { m = m shl 1; e-- }
                    m = m and 0x3FF
                    Float.fromBits(sign or ((e + 127) shl 23) or (m shl 13))
                }
            }
            31 -> Float.fromBits(sign or (0xFF shl 23) or (mant shl 13))
            else -> Float.fromBits(sign or ((exp - 15 + 127) shl 23) or (mant shl 13))
        }
    }
}
