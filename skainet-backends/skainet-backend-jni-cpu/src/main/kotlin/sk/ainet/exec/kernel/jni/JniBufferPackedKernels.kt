package sk.ainet.exec.kernel.jni

import java.nio.ByteBuffer
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelKey
import sk.ainet.backend.api.kernel.LayoutClass
import sk.ainet.backend.api.kernel.MappedCapableKernel
import sk.ainet.backend.api.kernel.OperandKey
import sk.ainet.backend.api.kernel.ReferenceMatmulKernel
import sk.ainet.backend.api.kernel.ViewKernel
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.DirectBufferStorage
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.MappedBufferStorage
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A [ViewKernel] over the JNI **row-major** matmuls, polymorphic in the weight's storage
 * (#1189/#1193): a `BLOCKED_ROW_MAJOR` packed weight is served wherever its bytes live —
 * off-heap ([MappedBufferStorage] mmap'd GGUF pages, [DirectBufferStorage]) through the
 * direct-buffer entries, or a heap `ByteArray` through the pinned-array entries. Same C kernels
 * either way; the pointer's provenance is a bridge detail, not a kernel property.
 *
 * The heap arm is what makes canonical (un-prepacked) heap weights fast: before it existed, a
 * heap tensor in file order fell to the decoding reference silently — measured 48,771 ms/step
 * vs 65 on a mixed-quant model (#1193). The remaining fallbacks (non-contiguous activation,
 * exotic storage) now announce themselves to the trace sink instead of hiding.
 *
 * Contrast with [sk.ainet.backend.api.kernel.PackedViewMatmulKernel], which serves the same
 * encodings from heap bytes in `BLOCKED_INPUT_MAJOR` (prepacked feed) order.
 */
public class JniRowMajorMatmulKernel(
    encodingName: String,
    override val key: KernelKey,
    private val bufferMatmul: (
        input: FloatArray, inputOffset: Int,
        weight: ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) -> Unit,
    private val arrayMatmul: (
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) -> Unit,
) : ViewKernel, MappedCapableKernel {

    override val name: String = "jni-rowmajor-$encodingName"

    override fun run(inputs: List<TensorView>, out: TensorView): Unit = run(inputs, out, NoopTraceSink)

    override fun run(inputs: List<TensorView>, out: TensorView, sink: TraceSink) {
        require(inputs.size == 2) { "matmul takes two operands" }
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions disagree: [$rows, $k] × [$n, ${w.shape[1]}]" }
        check(w.layout.blockOrder == BlockOrder.ROW_MAJOR) {
            "$name reads canonical row-major weights; this one is ${w.layout.blockOrder}"
        }

        val aHeap = a.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "activation storage ${a.storage::class.simpleName}")
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out, sink, "output storage ${out.storage::class.simpleName}")
        val activation = aHeap.floats ?: return fallback(inputs, out, sink, "activation is not a FloatArray")
        val output = oHeap.floats ?: return fallback(inputs, out, sink, "output is not a FloatArray")
        // The JNI kernel takes a contiguous activation row; a strided one would be mis-indexed.
        if (!a.isContiguous) return fallback(inputs, out, sink, "strided activation")

        val weightOffset = (w.layout.offsetElements * w.layout.elementBytes).toInt()
        when (val s = w.storage) {
            is MappedBufferStorage, is DirectBufferStorage -> {
                val buffer = if (s is MappedBufferStorage) s.buffer() else (s as DirectBufferStorage).buffer()
                for (r in 0 until rows) {
                    bufferMatmul(
                        activation, aHeap.arrayOffset + (a.layout.offsetElements + r.toLong() * k).toInt(),
                        buffer, weightOffset,
                        k, n,
                        output, oHeap.arrayOffset + (out.layout.offsetElements + r.toLong() * n).toInt(),
                    )
                }
            }
            is Storage.Heap -> {
                val bytes = s.bytes ?: return fallback(inputs, out, sink, "heap weight is not a ByteArray")
                for (r in 0 until rows) {
                    arrayMatmul(
                        activation, aHeap.arrayOffset + (a.layout.offsetElements + r.toLong() * k).toInt(),
                        bytes, s.arrayOffset + weightOffset,
                        k, n,
                        output, oHeap.arrayOffset + (out.layout.offsetElements + r.toLong() * n).toInt(),
                    )
                }
            }
            else -> return fallback(inputs, out, sink, "weight storage ${s::class.simpleName}")
        }
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView, sink: TraceSink, reason: String) {
        // #1193: never silent — the decoding reference is ~1000× the packed kernel, and that has
        // to be a line in the trace, exactly the way an inserted adapter is (SKEEP-003 §5.1).
        if (sink.isEnabled) {
            sink.emit(
                TraceEvent.KernelRun(
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
        /** The key this kernel serves: dense contiguous FP32 activation × row-major packed weight. */
        public fun keyFor(encoding: TensorEncoding): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, encoding), LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )
    }
}

/**
 * Registers the row-major kernels (#1189/#1192/#1193) into [KernelDispatch]: all seven GGML block formats, each serving mapped, direct and heap weights in canonical file order. Call next to
 * `KernelPacks.install(JniKernelProvider)` on Android — without it a canonical packed weight
 * (mapped OR un-prepacked heap) falls back to the decoding reference.
 */
public object JniMappedKernelPack {
    public fun install() {
        if (!JniKernelProvider.isAvailable()) return
        fun register(
            encoding: TensorEncoding,
            buffer: (FloatArray, Int, ByteBuffer, Int, Int, Int, FloatArray, Int) -> Unit,
            array: (FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit,
        ) = KernelDispatch.register(
            JniRowMajorMatmulKernel(encoding.name, JniRowMajorMatmulKernel.keyFor(encoding), buffer, array),
        )
        register(TensorEncoding.Q4_K, JniKernels::q4kMatmulRmDirect, JniKernels::q4kMatmulRm)
        register(TensorEncoding.Q6_K, JniKernels::q6kMatmulRmDirect, JniKernels::q6kMatmulRm)
        register(TensorEncoding.Q5_K, JniKernels::q5kMatmulRmDirect, JniKernels::q5kMatmulRm)
        register(TensorEncoding.Q8_0, JniKernels::q80MatmulRmDirect, JniKernels::q80MatmulRm)
        register(TensorEncoding.Q4_0, JniKernels::q40MatmulRmDirect, JniKernels::q40MatmulRm)
        register(TensorEncoding.Q5_0, JniKernels::q50MatmulRmDirect, JniKernels::q50MatmulRm)
        register(TensorEncoding.Q5_1, JniKernels::q51MatmulRmDirect, JniKernels::q51MatmulRm)
    }
}
