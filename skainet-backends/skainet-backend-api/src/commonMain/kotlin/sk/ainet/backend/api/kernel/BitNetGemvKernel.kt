package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.isTernary
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The reference `bitnet_gemv` (SKEEP-003 §5.3, M2-F3): int8 activations against ternary weights,
 * with **no multiplies in the inner loop**.
 *
 * A ternary weight is `-1`, `0` or `+1` times a scale, so the dot product is an add, a subtract or
 * nothing at all — the whole point of 1.58-bit weights, and the shape a NEON kernel (#1041) will
 * take. This one is plain Kotlin and runs everywhere: it is the thing the SIMD versions are
 * checked against, so it is written for clarity, and the scales are factored out of the loop
 * exactly as the vector kernels will factor them.
 *
 * Operands: `[rows, k]` activations in [I8Absmax.FORMAT] × `[n, k]` ternary weights in canonical
 * ([sk.ainet.lang.memory.BlockOrder.ROW_MAJOR]) block order — the order [TernaryCodec] produces and GGUF stores. Output `[rows, n]`.
 *
 * The weight's codes are read once per call, not once per row: a decode step is one row against
 * the whole matrix, so hoisting it is the difference between O(rows·n·k) decodes and O(n·k).
 */
public class BitNetGemvKernel(override val key: KernelKey) : ViewKernel {

    override val name: String get() = "bitnet_gemv/reference"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "bitnet_gemv takes (activation, weight), got ${inputs.size} operands" }
        val a = inputs[0]
        val w = inputs[1]
        require(a.format == I8Absmax.FORMAT) { "activation must be ${I8Absmax.FORMAT}, was ${a.format}" }
        require(w.format.encoding.isTernary) { "weight must be ternary, was ${w.format}" }
        require(a.shape.rank == 2 && w.shape.rank == 2 && out.shape.rank == 2) { "bitnet_gemv is 2-D" }
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions differ: activation k=$k, weight k=${w.shape[1]}" }
        require(out.shape[0] == rows && out.shape[1] == n) { "out must be [$rows, $n], was ${out.shape}" }

        val encoding = w.format.encoding
        val bytes = weightBytes(w)
        val codes = TernaryCodec.codes(encoding, bytes, n * k)
        val blockSize = blockSizeOf(encoding, n * k)

        for (r in 0 until rows) {
            val activation = I8Absmax.rowCodes(a, r)
            val activationScale = I8Absmax.scaleOf(a, r)
            for (o in 0 until n) {
                var acc = 0f
                var index = o * k
                var offset = 0
                // Walk the row block by block: within a block the weight scale is constant, so the
                // inner loop only adds and subtracts activation codes (and skips the zeros).
                while (offset < k) {
                    val run = minOf(blockSize - (index % blockSize), k - offset)
                    var partial = 0
                    for (i in 0 until run) {
                        when (codes[index + i].toInt()) {
                            1 -> partial += activation[offset + i].toInt()
                            -1 -> partial -= activation[offset + i].toInt()
                            else -> Unit                       // zero weights cost nothing
                        }
                    }
                    acc += partial * blockScale(encoding, bytes, (index + run - 1) / blockSize)
                    index += run
                    offset += run
                }
                out.set(r, o, value = acc * activationScale)
            }
        }
    }

    private fun weightBytes(w: TensorView): ByteArray = when (val storage = w.storage) {
        is Storage.Heap -> storage.bytes ?: throw UnsupportedOperationException("ternary weights need byte storage")
        // Off-heap/mapped weights (#1202): a transient snapshot, not a standing heap copy — this
        // reference kernel is the slow/fallback path already, so the extra copy here doesn't
        // regress the fast native path's residency.
        else -> ByteArray(storage.sizeBytes.toInt()).also { storage.copyInto(it) }
    }

    /** Elements per scale: a GGML block, or the whole tensor for the per-tensor BitNet encoding. */
    private fun blockSizeOf(encoding: TensorEncoding, elements: Int): Int =
        when (encoding) {
            TensorEncoding.TQ1_0 -> TensorEncoding.TQ1_0.BLOCK_SIZE
            TensorEncoding.TQ2_0 -> TensorEncoding.TQ2_0.BLOCK_SIZE
            else -> elements
        }

    /** The scale of block [block] — per-block FP16 for the GGML types, one FP32 for BitNet. */
    private fun blockScale(encoding: TensorEncoding, bytes: ByteArray, block: Int): Float = when (encoding) {
        TensorEncoding.TQ1_0 -> fp16At(bytes, block * TensorEncoding.TQ1_0.BYTES_PER_BLOCK + 52)
        TensorEncoding.TQ2_0 -> fp16At(bytes, block * TensorEncoding.TQ2_0.BYTES_PER_BLOCK + 64)
        else -> TernaryCodec.bitNetScale(bytes, (bytes.size - TensorEncoding.BITNET_B1_58.SCALE_BYTES) * 4)
    }

    private fun fp16At(bytes: ByteArray, offset: Int): Float =
        sk.ainet.lang.types.Fp16Codec.decode(
            (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8),
        )

    public companion object {
        /** The key this kernel serves for a ternary [weightFormat]. */
        public fun keyFor(weightFormat: Format): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(I8Absmax.FORMAT),
                OperandKey(weightFormat, LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )

        /** Register the reference kernel for every ternary encoding that carries its own bytes. */
        public fun registerReference() {
            for (encoding in listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0, TensorEncoding.BITNET_B1_58)) {
                val format = Format(sk.ainet.lang.types.FP32, encoding)
                KernelDispatch.register(BitNetGemvKernel(keyFor(format)))
            }
        }
    }
}
