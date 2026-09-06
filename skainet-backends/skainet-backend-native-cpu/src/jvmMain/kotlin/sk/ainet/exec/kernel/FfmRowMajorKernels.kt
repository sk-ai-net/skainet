package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
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
 * The JVM (FFM) row-major tier (#1191): serves `BLOCKED_ROW_MAJOR` packed weights through the
 * same `_rm` C symbols the Android JNI tier uses (#1189/#1192), polymorphic in the weight's
 * storage:
 *
 * - [MappedBufferStorage]/[DirectBufferStorage] → `MemorySegment.ofBuffer` — **zero-copy**: the
 *   kernel reads the mmap'd GGUF pages directly, which is the point of `WeightResidency.MAPPED`.
 *   Before this tier, a mapped packed weight on the JVM fell to the decoding reference kernel.
 * - Heap `ByteArray` → staged into a confined arena per call, the same cost class as the
 *   existing heap FFM kernels ([NativeQ4KMatmulKernel] et al. stage their weight too).
 *
 * Activations and outputs are heap `FloatArray`s staged per call (small: `k` and `n` floats).
 * Fallbacks to the decoding reference are announced to the trace sink (#1193) — never silent.
 */
public class FfmRowMajorMatmulKernel internal constructor(
    encodingName: String,
    override val key: KernelKey,
    private val handle: MethodHandle,
) : ViewKernel, MappedCapableKernel {

    override val name: String = "ffm-rowmajor-$encodingName"

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
        if (!a.isContiguous) return fallback(inputs, out, sink, "strided activation")

        val weightOffset = (w.layout.offsetElements * w.layout.elementBytes).toInt()

        Arena.ofConfined().use { arena ->
            val weightSeg: MemorySegment
            val weightSegOffset: Int
            when (val s = w.storage) {
                is MappedBufferStorage, is DirectBufferStorage -> {
                    val buffer = if (s is MappedBufferStorage) s.buffer() else (s as DirectBufferStorage).buffer()
                    weightSeg = MemorySegment.ofBuffer(buffer)   // zero-copy over the mapping
                    weightSegOffset = weightOffset
                }
                is Storage.Heap -> {
                    val bytes = s.bytes ?: return fallback(inputs, out, sink, "heap weight is not a ByteArray")
                    val length = s.sizeBytes - weightOffset
                    val seg = arena.allocate(length, 1L)
                    MemorySegment.copy(bytes, s.arrayOffset + weightOffset, seg, ValueLayout.JAVA_BYTE, 0L, length.toInt())
                    weightSeg = seg
                    weightSegOffset = 0
                }
                else -> return fallback(inputs, out, sink, "weight storage ${s::class.simpleName}")
            }

            val inSeg = arena.allocate(k.toLong() * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())
            val outSeg = arena.allocate(n.toLong() * java.lang.Float.BYTES, ValueLayout.JAVA_FLOAT.byteAlignment())
            for (r in 0 until rows) {
                MemorySegment.copy(
                    activation, aHeap.arrayOffset + (a.layout.offsetElements + r.toLong() * k).toInt(),
                    inSeg, ValueLayout.JAVA_FLOAT, 0L, k,
                )
                handle.invoke(inSeg, 0, weightSeg, weightSegOffset, k, n, outSeg, 0)
                MemorySegment.copy(
                    outSeg, ValueLayout.JAVA_FLOAT, 0L,
                    output, oHeap.arrayOffset + (out.layout.offsetElements + r.toLong() * n).toInt(), n,
                )
            }
        }
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView, sink: TraceSink, reason: String) {
        // #1193: the decoding reference is ~1000× slower — that must be a trace line, never silent.
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
 * Registers the FFM row-major kernels (#1191) into [KernelDispatch] — the JVM counterpart of
 * Android's `JniMappedKernelPack.install()`. No-op when the native library is unavailable;
 * a symbol that fails to bind simply leaves that encoding to the reference kernel.
 */
public object FfmRowMajorKernelPack {

    private val DESCRIPTOR = FunctionDescriptor.ofVoid(
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // input, input_offset
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // weight, weight_byte_offset
        ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,  // input_dim, output_dim
        ValueLayout.ADDRESS, ValueLayout.JAVA_INT,   // output, output_offset
    )

    private val SYMBOLS: List<Pair<TensorEncoding, String>> = listOf(
        TensorEncoding.Q4_K to "skainet_q4k_matmul_rm",
        TensorEncoding.Q6_K to "skainet_q6k_matmul_rm",
        TensorEncoding.Q5_K to "skainet_q5k_matmul_rm",
        TensorEncoding.Q8_0 to "skainet_q8_0_matmul_rm",
        TensorEncoding.Q4_0 to "skainet_q4_0_matmul_rm",
        TensorEncoding.Q5_0 to "skainet_q5_0_matmul_rm",
        TensorEncoding.Q5_1 to "skainet_q5_1_matmul_rm",
    )

    public fun install() {
        val lookup = NativeLibraryLoader.lookup() ?: return
        for ((encoding, symbol) in SYMBOLS) {
            val sym = lookup.find(symbol).orElse(null) ?: continue
            val handle = runCatching { Linker.nativeLinker().downcallHandle(sym, DESCRIPTOR) }.getOrNull() ?: continue
            KernelDispatch.register(
                FfmRowMajorMatmulKernel(encoding.name, FfmRowMajorMatmulKernel.keyFor(encoding), handle),
            )
        }
    }
}
