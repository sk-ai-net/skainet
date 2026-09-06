package sk.ainet.io.gguf

import sk.ainet.io.weights.NameMap
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.plan.KvCacheMode
import sk.ainet.lang.memory.plan.ModelGeometry
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.PlanTensor
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int64
import sk.ainet.lang.types.Int8

/**
 * Build a [PlanInput] from a GGUF **header only** — tensor names, shapes and types plus the
 * architecture metadata keys; no tensor bytes are read (PRD M0-F1). [ctx] defaults to the trained
 * context length of the model when the header has one.
 */
public fun StreamingGGUFReader.planInput(
    ctx: Int? = null,
    prefillChunk: Int = PlanInput.DEFAULT_PREFILL_CHUNK,
    kvMode: KvCacheMode = KvCacheMode.BF16,
    nameMap: NameMap? = nameMap(),
    /**
     * The [sk.ainet.lang.memory.plan.WeightForm] the load will use, per tensor name — the same
     * knob `StreamingGgufParametersLoader` takes, so the plan prices what the load will actually
     * do (#1189: under `MAPPED` the servable encodings are budgeted against the page cache, not
     * the heap). `null` for every tensor plans the pre-form default: everything heap-charged.
     */
    formFor: (String) -> sk.ainet.lang.memory.plan.WeightForm? = { null },
): PlanInput {
    val arch = fields["general.architecture"] as? String ?: "unknown"
    val name = fields["general.name"] as? String ?: arch
    val geometry = ggufGeometry(arch)
    val weights = tensors.map { t ->
        val format = ggufFormat(t.tensorType, t.nBytes)
        PlanTensor(
            name = t.name,
            id = nameMap?.toTensorId(t.name),
            format = format,
            elementCount = t.nElements,
            bytes = format.physicalBytes(t.nElements) ?: t.nBytes,
            form = formFor(t.name),
        )
    }
    val ctxUsed = ctx ?: geometry?.trainedContextLength ?: 2048
    return PlanInput(name, arch, weights, geometry, ctxUsed, prefillChunk, kvMode)
}

/** Architecture metadata (`<arch>.block_count`, `<arch>.attention.head_count`, …) as a [ModelGeometry], or `null` if the header lacks it. */
public fun StreamingGGUFReader.ggufGeometry(architecture: String? = fields["general.architecture"] as? String): ModelGeometry? {
    val arch = architecture ?: return null
    fun int(key: String): Int? = (fields["$arch.$key"] as? Number)?.toInt() ?: (fields["$arch.$key"] as? UInt)?.toInt()
    val layers = int("block_count") ?: return null
    val emb = int("embedding_length") ?: return null
    val heads = int("attention.head_count") ?: return null
    val kvHeads = int("attention.head_count_kv") ?: heads
    val headDim = int("attention.key_length") ?: (emb / heads)
    val valueDim = int("attention.value_length") ?: headDim
    val ffn = int("feed_forward_length") ?: (4 * emb)
    val vocab = int("vocab_size") ?: (fields["tokenizer.ggml.tokens"] as? List<*>)?.size
        ?: tensors.firstOrNull { it.name == "token_embd.weight" }?.shape?.lastOrNull()?.toInt() ?: 0
    return ModelGeometry(layers, heads, kvHeads, headDim, valueDim, emb, ffn, vocab, int("context_length"))
}

/** `Format` of a GGUF tensor type: quantized types are logically FP32 with their block encoding. */
public fun ggufFormat(type: GGMLQuantizationType, nBytes: Long): Format {
    val dtype: DType = when (type) {
        GGMLQuantizationType.F32 -> FP32; GGMLQuantizationType.F16 -> FP16; GGMLQuantizationType.BF16 -> BF16; GGMLQuantizationType.F64 -> FP64
        GGMLQuantizationType.I8 -> Int8; GGMLQuantizationType.I16 -> Int16; GGMLQuantizationType.I32 -> Int32; GGMLQuantizationType.I64 -> Int64
        else -> FP32
    }
    val encoding: TensorEncoding = when (type) {
        GGMLQuantizationType.F32 -> TensorEncoding.Dense(4); GGMLQuantizationType.F16, GGMLQuantizationType.BF16 -> TensorEncoding.Dense(2)
        GGMLQuantizationType.F64 -> TensorEncoding.Dense(8); GGMLQuantizationType.I8 -> TensorEncoding.Dense(1)
        GGMLQuantizationType.I16 -> TensorEncoding.Dense(2); GGMLQuantizationType.I32 -> TensorEncoding.Dense(4); GGMLQuantizationType.I64 -> TensorEncoding.Dense(8)
        GGMLQuantizationType.Q4_0 -> TensorEncoding.Q4_0; GGMLQuantizationType.Q5_0 -> TensorEncoding.Q5_0; GGMLQuantizationType.Q5_1 -> TensorEncoding.Q5_1
        GGMLQuantizationType.Q8_0 -> TensorEncoding.Q8_0; GGMLQuantizationType.Q4_K -> TensorEncoding.Q4_K; GGMLQuantizationType.Q5_K -> TensorEncoding.Q5_K
        GGMLQuantizationType.Q6_K -> TensorEncoding.Q6_K
        GGMLQuantizationType.TQ1_0 -> TensorEncoding.TQ1_0; GGMLQuantizationType.TQ2_0 -> TensorEncoding.TQ2_0
        else -> TensorEncoding.Opaque(type.name, nBytes)
    }
    return Format(dtype, encoding)
}
