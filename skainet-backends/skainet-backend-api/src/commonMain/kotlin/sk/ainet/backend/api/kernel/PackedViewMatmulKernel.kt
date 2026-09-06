package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A [ViewKernel] over one of the packed SPI matmul kernels (SKEEP-003 §5.2; #973.2, #1095).
 *
 * These kernels read their weight **input-block-major** — `(blockIdx * outputDim + o)`, every
 * output row's block for one input block contiguous — which is why bridging them was deferred in
 * #1029: a packed `TensorView` loaded from a file is canonical row-major, and nothing in the key
 * said so. Now it does ([BlockOrder]), so the kernel declares the order it reads and the dispatcher
 * relayouts when the operand disagrees.
 *
 * One row at a time: the SPI is a matrix-vector kernel, so an activation of `m` rows is `m` calls,
 * which is what the decode path does anyway (`m == 1`).
 */
public class PackedViewMatmulKernel(
    providerName: String,
    private val encodingName: String,
    override val key: KernelKey,
    private val matmul: (
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) -> Unit,
) : ViewKernel {

    override val name: String = "$providerName-$encodingName"

    override fun run(inputs: List<TensorView>, out: TensorView): Unit =
        run(inputs, out, sk.ainet.lang.memory.trace.NoopTraceSink)

    override fun run(inputs: List<TensorView>, out: TensorView, sink: sk.ainet.lang.memory.trace.TraceSink) {
        require(inputs.size == 2) { "matmul takes two operands" }
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions disagree: [$rows, $k] × [$n, ${w.shape[1]}]" }
        check(w.layout.blockOrder == BlockOrder.INPUT_BLOCK_MAJOR) {
            "$name reads input-block-major weights; this one is ${w.layout.blockOrder} — the dispatcher " +
                "should have prepacked it (#973)"
        }

        val aHeap = a.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "activation storage ${a.storage::class.simpleName}")
        val wHeap = w.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "weight storage ${w.storage::class.simpleName} — feed-order kernels take heap bytes")
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "output storage ${out.storage::class.simpleName}")
        val activation = aHeap.floats ?: return fallback(inputs, out, sink, "activation is not a FloatArray")
        val weight = wHeap.bytes ?: return fallback(inputs, out, sink, "weight is not a ByteArray")
        val output = oHeap.floats ?: return fallback(inputs, out, sink, "output is not a FloatArray")
        // The SPI takes a contiguous activation row; a strided one would be mis-indexed, so it goes
        // to the reference kernel rather than silently reading the wrong floats.
        if (!a.isContiguous) return fallback(inputs, out, sink, "strided activation")

        val weightOffset = wHeap.arrayOffset + (w.layout.offsetElements * w.layout.elementBytes).toInt()
        for (r in 0 until rows) {
            matmul(
                activation, aHeap.arrayOffset + (a.layout.offsetElements + r.toLong() * k).toInt(),
                weight, weightOffset,
                k, n,
                output, oHeap.arrayOffset + (out.layout.offsetElements + r.toLong() * n).toInt(),
            )
        }
    }

    private fun fallback(
        inputs: List<TensorView>,
        out: TensorView,
        sink: sk.ainet.lang.memory.trace.TraceSink,
        reason: String,
    ) {
        // #1193: the decoding reference is ~1000× slower — a trace line, never silent.
        if (sink.isEnabled) {
            sink.emit(
                sk.ainet.lang.memory.trace.TraceEvent.KernelRun(
                    op = key.op,
                    kernel = "reference-fallback from $name: $reason",
                    inputs = inputs.map { it.id },
                    output = out.id,
                ),
            )
        }
        ReferenceMatmulKernel(key).run(inputs, out)
    }

    public companion object {
        /** The key a packed kernel serves: a dense contiguous activation × an input-major weight. */
        public fun keyFor(encoding: TensorEncoding, capabilities: Set<String> = emptySet()): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, encoding), LayoutClass.BLOCKED_INPUT_MAJOR),
            ),
            capabilities = capabilities,
        )
    }
}
