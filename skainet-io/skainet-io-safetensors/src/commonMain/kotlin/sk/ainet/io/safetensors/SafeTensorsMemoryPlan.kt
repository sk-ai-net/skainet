package sk.ainet.io.safetensors

import sk.ainet.lang.memory.Format
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
 * Build a [PlanInput] from a safetensors **header only** — the JSON tensor table at the start of
 * the file; no tensor payload is read (#1169).
 *
 * safetensors carries no architecture metadata, so [PlanInput.geometry] is `null` and the plan is
 * weights-only: KV cache and forward slab are not modelled, and the caller should say so.
 * Per-tensor byte counts come from the header's `data_offsets` — authoritative even for dtypes
 * with no fixed per-element width.
 */
public fun StreamingSafeTensorsReader.planInput(
    modelName: String,
    ctx: Int = 1,
): PlanInput {
    val weights = tensors.map { t ->
        PlanTensor(
            name = t.name,
            id = null,
            format = safeTensorsFormat(t.dtype, t.sizeInBytes),
            elementCount = t.elementCount,
            bytes = t.sizeInBytes,
        )
    }
    return PlanInput(
        modelName = modelName,
        architecture = "safetensors",
        weights = weights,
        geometry = null,
        ctx = ctx,
    )
}

/** The [Format] a safetensors dtype string describes; unknown dtypes become an opaque encoding priced by [sizeInBytes]. */
public fun safeTensorsFormat(dtype: String, sizeInBytes: Long): Format {
    val known: Pair<DType, Int>? = when (dtype) {
        "F32" -> FP32 to 4
        "F16" -> FP16 to 2
        "BF16" -> BF16 to 2
        "F64" -> FP64 to 8
        "I8", "U8", "BOOL" -> Int8 to 1
        "I16", "U16" -> Int16 to 2
        "I32", "U32" -> Int32 to 4
        "I64", "U64" -> Int64 to 8
        else -> null
    }
    return if (known != null) {
        Format(known.first, TensorEncoding.Dense(known.second))
    } else {
        Format(FP32, TensorEncoding.Opaque(dtype, sizeInBytes))
    }
}
