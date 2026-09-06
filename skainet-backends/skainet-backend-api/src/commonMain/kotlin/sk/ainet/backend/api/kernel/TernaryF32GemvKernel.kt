package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * The reference `ternary_f32_gemv` (#1138): **FP32 activations** against `BITNET_B1_58` ternary
 * weights — exact, no activation quantization.
 *
 * This is the f32 sibling of [BitNetGemvKernel]. That kernel is the W1.58A8 path: it asks the
 * dispatcher to requantize activations to int8-absmax first, trading ~1.5 % quantization error for
 * `sdot`-friendly integer math. This one keeps the activations as they are — a ternary weight is
 * a `+1`, `-1` or nothing, so the dot product is float adds and subtracts, and the result equals
 * the FP32 matmul against the decoded weight bit-for-bit apart from summation order. It serves the
 * **exact** dispatch key (FP32 dense × `BITNET_B1_58`), which `KernelDispatch.matmul` checks
 * *before* the requantize branch — installing this pack short-circuits the int8 adapter with zero
 * dispatcher changes.
 *
 * Only `BITNET_B1_58` is served: its payload (four codes per byte, low bit-pair first, in element
 * order) is what the vendored NeoGPU LUT kernel (#1137) reads, and its scale is in-band. The GGML
 * block types (TQ1_0/TQ2_0) have per-block scales and interleaved payloads — they keep their
 * int8 path.
 *
 * Operands: `[rows, k]` FP32 dense activations × `[n, k]` `BITNET_B1_58` weights in canonical
 * row-major order. Output `[rows, n]` FP32. The per-tensor scale is applied to the output —
 * native implementations ([TernaryF32GemvNative]) do not see it.
 */
public class TernaryF32GemvKernel(override val key: KernelKey) : ViewKernel {

    override val name: String get() = "ternary_f32_gemv/reference"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "ternary_f32_gemv takes (activation, weight), got ${inputs.size} operands" }
        val a = inputs[0]
        val w = inputs[1]
        require(a.format.dtype == FP32) { "activation must be FP32, was ${a.format}" }
        require(w.format.encoding == TensorEncoding.BITNET_B1_58) {
            "weight must be ${TensorEncoding.BITNET_B1_58}, was ${w.format}"
        }
        require(a.shape.rank == 2 && w.shape.rank == 2 && out.shape.rank == 2) { "ternary_f32_gemv is 2-D" }
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions differ: activation k=$k, weight k=${w.shape[1]}" }
        require(out.shape[0] == rows && out.shape[1] == n) { "out must be [$rows, $n], was ${out.shape}" }
        if (rows == 0 || n == 0) return

        val (bytes, byteOffset) = weightBytes(w)
        // Codes are hoisted out of the row loop, and the per-tensor scale out of everything.
        val codes = TernaryCodec.codes(TensorEncoding.BITNET_B1_58, bytes, n * k, byteOffset)
        val scale = TernaryCodec.bitNetScale(bytes, n * k, byteOffset)

        for (r in 0 until rows) {
            for (o in 0 until n) {
                var acc = 0f
                val base = o * k
                for (i in 0 until k) {
                    when (codes[base + i].toInt()) {
                        1 -> acc += a.get(r, i)
                        -1 -> acc -= a.get(r, i)
                        2 -> acc += 2f * a.get(r, i)   // byte code 3; loaders reject it, decode agrees
                        else -> Unit                    // zero weights cost nothing
                    }
                }
                out.set(r, o, value = acc * scale)
            }
        }
    }

    /** The weight's bytes and the byte offset codes/scale should be read from within them. */
    private fun weightBytes(w: TensorView): Pair<ByteArray, Int> = when (val storage = w.storage) {
        is Storage.Heap -> (storage.bytes ?: throw UnsupportedOperationException("ternary weights need byte storage")) to storage.arrayOffset
        // Off-heap/mapped weights (#1202): a transient snapshot, not a standing heap copy — this
        // reference kernel is the slow/fallback path already, so the extra copy here doesn't
        // regress the fast native path's residency. The snapshot is tensor-sized, so the codec
        // reads it from offset 0 regardless of where the storage itself lives.
        else -> ByteArray(storage.sizeBytes.toInt()).also { storage.copyInto(it) } to 0
    }

    public companion object {
        /** The exact key this kernel serves: FP32 dense contiguous × `BITNET_B1_58` row-major. */
        public fun keyFor(): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, TensorEncoding.BITNET_B1_58), LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )
    }
}
