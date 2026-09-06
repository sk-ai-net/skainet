package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.TensorView

/**
 * A registered kernel: a function `(inputs, out) -> Unit` behind a [KernelKey] (SKEEP-003 §0
 * *Kernel*, §5.2). Custom kernels are written against views and registered — an author never
 * touches a `TensorData` subclass.
 */
public interface ViewKernel {
    /** What this kernel serves. */
    public val key: KernelKey

    /** A name for logs, traces and `UnsupportedKernel` messages (`scalar-reference`, `panama-q4k`, …). */
    public val name: String

    /** Run the kernel: [inputs] as described by [key], result written into [out]. */
    public fun run(inputs: List<TensorView>, out: TensorView)

    /**
     * Run with a [sink] for events the kernel itself must report — above all an internal
     * fallback to the decoding reference (#1193: a silent 1000× slowdown deserves a trace
     * line the way an adapter gets one, SKEEP-003 §5.1). Default delegates to [run]; kernels
     * with an internal fallback override this and emit before punting.
     */
    public fun run(inputs: List<TensorView>, out: TensorView, sink: sk.ainet.lang.memory.trace.TraceSink): Unit =
        run(inputs, out)
}

/**
 * The reference matmul: correct for **any** pair of formats and layouts, because it reads through
 * `TensorView.get()`, which decodes (rule 4). Slow by design — it is the fallback that makes an
 * unsupported combination produce right numbers with a warning instead of a `ClassCastException`
 * in layer 17 (#993). Registered for every key the dispatcher cannot serve better.
 *
 * `out = a × bᵀ` in the shapes SKaiNET's dispatch normalises to: `a` is `[m, k]`, `b` is `[n, k]`
 * (a weight stored output-major, as GGUF does), `out` is `[m, n]`.
 */
public class ReferenceMatmulKernel(override val key: KernelKey) : ViewKernel {
    override val name: String get() = "reference"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "matmul takes two operands, got ${inputs.size}" }
        val a = inputs[0]; val b = inputs[1]
        require(a.shape.rank == 2 && b.shape.rank == 2 && out.shape.rank == 2) { "reference matmul works on rank-2 views (normalise first)" }
        val m = a.shape[0]; val k = a.shape[1]; val n = b.shape[0]
        require(b.shape[1] == k) { "inner dimensions disagree: a is [${m}, ${k}], b is [${n}, ${b.shape[1]}]" }
        require(out.shape[0] == m && out.shape[1] == n) { "out must be [$m, $n], was ${out.shape}" }
        for (i in 0 until m) {
            for (j in 0 until n) {
                var acc = 0f
                for (t in 0 until k) acc += a.get(i, t) * b.get(j, t)
                out.set(i, j, value = acc)
            }
        }
    }

    public companion object {
        /** The reference kernel for the formats of [a] and [b]. */
        public fun forOperands(a: Format, b: Format): ReferenceMatmulKernel =
            ReferenceMatmulKernel(KernelKey("matmul", listOf(OperandKey.contiguous(a), OperandKey.contiguous(b))))
    }
}
