package sk.ainet.io.onnx

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
 * Build a [PlanInput] from an ONNX model's **initializer table only** — the streaming reader
 * records names, dims, dtypes and byte counts (including `external_data` lengths for >2 GB
 * models) without materializing any tensor (#1169).
 *
 * ONNX carries no transformer-geometry metadata, so [PlanInput.geometry] is `null` and the plan
 * is weights-only: KV cache and forward slab are not modelled, and the caller should say so.
 */
public fun StreamingOnnxReader.planInput(
    modelName: String,
    ctx: Int = 1,
): PlanInput {
    val weights = tensors.map { t ->
        PlanTensor(
            name = t.name,
            id = null,
            format = onnxFormat(t.dataType, t.dataTypeName, t.estimatedBytesLong),
            elementCount = t.nElements,
            bytes = t.estimatedBytesLong,
        )
    }
    return PlanInput(
        modelName = modelName,
        architecture = "onnx",
        weights = weights,
        geometry = null,
        ctx = ctx,
    )
}

/** The [Format] an ONNX `TensorProto.DataType` describes; unknown types become an opaque encoding priced by [sizeInBytes]. */
public fun onnxFormat(dataType: Int, dataTypeName: String, sizeInBytes: Long): Format {
    val known: Pair<DType, Int>? = when (dataType) {
        1 -> FP32 to 4      // FLOAT
        10 -> FP16 to 2     // FLOAT16
        16 -> BF16 to 2     // BFLOAT16
        11 -> FP64 to 8     // DOUBLE
        2, 3, 9 -> Int8 to 1   // UINT8, INT8, BOOL
        4, 5 -> Int16 to 2  // UINT16, INT16
        6, 12 -> Int32 to 4 // INT32, UINT32
        7, 13 -> Int64 to 8 // INT64, UINT64
        else -> null
    }
    return if (known != null) {
        Format(known.first, TensorEncoding.Dense(known.second))
    } else {
        Format(FP32, TensorEncoding.Opaque(dataTypeName, sizeInBytes))
    }
}
