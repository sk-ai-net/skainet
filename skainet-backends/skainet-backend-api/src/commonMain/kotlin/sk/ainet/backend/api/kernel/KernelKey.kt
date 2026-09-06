package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * What a kernel consumes, declared rather than discovered (SKEEP-003 §0 *KernelKey*, §5.1): the op,
 * the [Format] of each operand, how each operand is laid out, and the placement it needs. The
 * dispatcher looks a key up instead of walking an `is`-ladder over `TensorData` subclasses — which
 * is what made every quantisation bug a dispatch bug (#993, #991).
 *
 * Keys are values: equal keys select the same kernel, and a key prints as something a log or an
 * `UnsupportedKernel` message can show: `matmul(F32/Dense(4B) contiguous × F32/Q4_K blocked) @host`.
 */
public data class KernelKey(
    val op: String,
    val operands: List<OperandKey>,
    val placement: Placement = Placement.HOST,
    /**
     * Platform capabilities a kernel requires (`vector`, `dotprod`, `i8mm`, `ffm`, …). Empty means
     * "no special requirement" — the portable kernel. A pack registers its key *with* the
     * capabilities it needs so a device that lacks them never selects it (§5.2, #920).
     */
    val capabilities: Set<String> = emptySet(),
) {
    /** Where the operands live — host memory today; a device backend adds its own (PRD non-goal for M1). */
    public enum class Placement { HOST, DEVICE }

    override fun toString(): String = buildString {
        append(op); append('('); append(operands.joinToString(" × ")); append(')')
        append(" @"); append(placement.name.lowercase())
        if (capabilities.isNotEmpty()) { append(" ["); append(capabilities.sorted().joinToString(",")); append(']') }
    }

    public companion object {
        /** The key of `matmul(activation, weight)` as the two views describe themselves. */
        public fun matmul(activation: TensorView, weight: TensorView, placement: Placement = Placement.HOST): KernelKey =
            KernelKey("matmul", listOf(OperandKey.of(activation), OperandKey.of(weight)), placement)
    }
}

/** One operand of a [KernelKey]: its [Format] plus the layout class the kernel must cope with. */
public data class OperandKey(val format: Format, val layout: LayoutClass) {
    override fun toString(): String = "$format ${layout.name.lowercase()}"

    public companion object {
        /** Describe [view]: dense-and-gap-free is `CONTIGUOUS`, a packed layout is `BLOCKED`, anything else `STRIDED`. */
        public fun of(view: TensorView): OperandKey {
            val cls = when {
                view.layout.blocked -> when (view.layout.blockOrder) {
                    sk.ainet.lang.memory.BlockOrder.ROW_MAJOR -> LayoutClass.BLOCKED_ROW_MAJOR
                    sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR -> LayoutClass.BLOCKED_INPUT_MAJOR
                }
                view.isContiguous -> LayoutClass.CONTIGUOUS
                else -> LayoutClass.STRIDED
            }
            return OperandKey(view.format, cls)
        }

        /** A dense contiguous operand of [format] — the shape kernels prefer. */
        public fun contiguous(format: Format): OperandKey = OperandKey(format, LayoutClass.CONTIGUOUS)
    }
}

/**
 * How an operand's bytes are arranged, as far as kernel selection cares: one gap-free run
 * ([CONTIGUOUS]), a strided view over a larger buffer ([STRIDED]), or block-packed in one of the
 * two orders that exist ([BLOCKED_ROW_MAJOR], [BLOCKED_INPUT_MAJOR]).
 *
 * A kernel that declares `CONTIGUOUS` gets a gather adapter inserted for a `STRIDED` operand
 * (§5.1) — the adapter is visible in the trace, never hidden inside a kernel. The two blocked
 * classes exist for the same reason: a packed kernel reads its weight in *one* of the two block
 * orders, and #973 is what happens when that is left implicit. A kernel declares which one it
 * takes, and the dispatcher relayouts when the operand disagrees.
 */
public enum class LayoutClass {
    CONTIGUOUS,
    STRIDED,

    /** Blocks in file order: `o * blocksPerRow + b` ([sk.ainet.lang.memory.BlockOrder.ROW_MAJOR]). */
    BLOCKED_ROW_MAJOR,

    /** Blocks in kernel feed order: `b * outputDim + o` ([sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR]). */
    BLOCKED_INPUT_MAJOR,
    ;

    /** True for either blocked class — when the question is "packed or not". */
    public val isBlocked: Boolean get() = this == BLOCKED_ROW_MAJOR || this == BLOCKED_INPUT_MAJOR
}

/** Thrown when no registered kernel and no adapter chain can serve a key; lists what is registered. */
public class UnsupportedKernelException(
    public val key: KernelKey,
    public val candidates: List<String>,
    message: String = "No kernel for $key" + if (candidates.isEmpty()) "" else "; registered: ${candidates.joinToString(", ")}",
) : IllegalArgumentException(message)

/** The encoding name a [KernelKey] uses for a format, matching `KernelProvider.supports`' dtype keys. */
public val Format.kernelEncodingName: String
    get() = when (val e = encoding) {
        is TensorEncoding.Dense -> dtype.name
        else -> e.name
    }
