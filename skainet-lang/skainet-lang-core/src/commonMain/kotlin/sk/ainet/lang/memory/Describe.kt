@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.TensorStorage

/**
 * One-line debugger rendering of a tensor (SKEEP-003 §4.7, PRD M0-F5):
 * `TensorId · Format · Shape · storage kind · origin · scope · StorageId`.
 * Fields that only exist from milestone M1 on (scope, storage id) print as `—`.
 *
 * Example: `model.layers.blk.3.attn.q_proj.weight · Float32/Q4_K · [2048, 2048] · Q4_KBlockTensorData · — · scope — · storage —`
 */
public fun Tensor<*, *>.describe(): String = buildString {
    append(id?.canonical ?: "—"); append(SEP)
    append(formatOrNull?.toString() ?: "?"); append(SEP)
    append(shapeText(data.shape.dimensions)); append(SEP)
    append(data::class.simpleName ?: "data"); append(SEP)
    append("—"); append(SEP)          // origin (file/offset) — storage-backed tensors only, M1
    append("scope —"); append(SEP)
    append("storage —")
}

/** Same rendering for a storage descriptor; origin is the file for file-backed buffers. */
public fun TensorStorage.describe(id: sk.ainet.lang.tensor.TensorId? = null): String = buildString {
    append(id?.canonical ?: "—"); append(SEP)
    append(format.toString()); append(SEP)
    append(shapeText(shape.dimensions)); append(SEP)
    append(
        when (val b = buffer) {
            is BufferHandle.Owned -> "Owned"
            is BufferHandle.Floats -> "Floats"
            is BufferHandle.Borrowed -> "Borrowed"
            is BufferHandle.Aliased -> "Aliased"
            is BufferHandle.FileBacked -> "Mapped"
            is BufferHandle.DeviceResident -> "Device(${b.deviceId})"
        }
    ); append(SEP)
    append(
        when (val b = buffer) {
            is BufferHandle.FileBacked -> "${b.path} @0x${b.fileOffset.toString(16)}"
            else -> "—"
        }
    ); append(SEP)
    append("scope —"); append(SEP)
    append("storage —")
}

private const val SEP = " · "

private fun shapeText(dims: IntArray): String = dims.joinToString(", ", "[", "]")
