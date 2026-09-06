package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A hand-written `bitnet_gemv` supplied by a platform pack (SKEEP-003 §5.2, M2-F4).
 *
 * Deliberately array-shaped rather than view-shaped: the implementations are JNI or cinterop shims
 * that pin primitive arrays, and keeping the SPI at that level means the pack modules carry no
 * knowledge of `TensorView`. [TernaryKernelPacks] does the unwrapping once.
 *
 * The weight is canonical row-major `TQ2_0` — the order [sk.ainet.lang.memory.TernaryCodec] writes
 * and a GGUF holds. That is *not* the block-major order the Q4_0…Q6_K SPI kernels take, which is
 * why this one can be bridged into the view registry while those still wait on #973: there is no
 * ambiguity here, and the parity test pins it.
 */
public interface BitNetGemvNative {
    /** A name for logs and traces, e.g. `neon-dotprod`. */
    public val name: String

    /** `out[o] = activationScale * Σ_k code(k) · weight(o, k)` for one token. */
    public fun gemvTq2_0(
        activation: ByteArray,
        activationOffset: Int,
        activationScale: Float,
        weight: ByteArray,
        weightByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
        out: FloatArray,
        outOffset: Int,
    )
}

/**
 * Installs the ternary kernels: the portable reference always, and a platform `bitnet_gemv` on top
 * of it when one is available (M2-F4).
 *
 * Removing the native artifact is not an error and never a crash — the reference kernel is already
 * registered, so dispatch keeps working at reference speed and the caller is *told* through [warn]
 * rather than left to wonder why decode got slower.
 */
public object TernaryKernelPacks {

    /** Capability a `bitnet_gemv` pack declares when it needs ARMv8.2 dot-product instructions. */
    public const val CAPABILITY_DOTPROD: String = "dotprod"

    private val ternaryEncodings = listOf(TensorEncoding.TQ1_0, TensorEncoding.TQ2_0, TensorEncoding.BITNET_B1_58)

    /**
     * @param native the platform kernel, or `null` when its artifact is absent
     * @param capabilities what [native] needs; recorded in the key so a device without them never
     *   selects it (§5.2)
     * @param warn where the "running without the native kernel" notice goes
     * @return the name of the kernel that will serve TQ2_0 — the pack's, or the reference
     */
    public fun install(
        native: BitNetGemvNative? = null,
        capabilities: Set<String> = emptySet(),
        warn: (String) -> Unit = {},
    ): String {
        for (encoding in ternaryEncodings) {
            KernelDispatch.register(BitNetGemvKernel(BitNetGemvKernel.keyFor(Format(FP32, encoding))))
        }
        if (native == null) {
            warn(
                "bitnet_gemv: no native kernel available — using the portable reference. " +
                    "Add the NEON artifact for the fast path; nothing else changes.",
            )
            return BitNetGemvKernel(BitNetGemvKernel.keyFor(Format(FP32, TensorEncoding.TQ2_0))).name
        }
        val key = BitNetGemvKernel.keyFor(Format(FP32, TensorEncoding.TQ2_0)).copy(capabilities = capabilities)
        val kernel = NativeBitNetGemvKernel(native, key)
        KernelDispatch.register(kernel)
        // Registering under the capability-free key too: the dispatcher builds its key from the
        // operands, which say nothing about the CPU. The pack only installs itself on a device that
        // *has* the capability, so the two keys select the same kernel there and neither exists on
        // a device that does not.
        KernelDispatch.register(NativeBitNetGemvKernel(native, BitNetGemvKernel.keyFor(Format(FP32, TensorEncoding.TQ2_0))))
        return kernel.name
    }
}

/**
 * A [ViewKernel] over a [BitNetGemvNative]: unwraps the two views once per call and hands the
 * native kernel the arrays it wants.
 *
 * Falls back to the reference for anything the native kernel does not take — a multi-row
 * activation (prefill) or storage that is not a heap array — instead of failing: the fast path is
 * an optimization, never a correctness requirement.
 */
public class NativeBitNetGemvKernel(
    private val native: BitNetGemvNative,
    override val key: KernelKey,
) : ViewKernel {

    override val name: String get() = "bitnet_gemv/${native.name}"

    private val reference = BitNetGemvKernel(key)

    override fun run(inputs: List<TensorView>, out: TensorView) {
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val activationBytes = (a.storage as? Storage.Heap)?.bytes
        val weightBytes = (w.storage as? Storage.Heap)?.bytes
        val outFloats = (out.storage as? Storage.Heap)?.floats
        if (rows != 1 || activationBytes == null || weightBytes == null || outFloats == null) {
            reference.run(inputs, out)
            return
        }
        native.gemvTq2_0(
            activation = activationBytes,
            activationOffset = (a.storage as Storage.Heap).arrayOffset,
            activationScale = I8Absmax.scaleOf(a, 0),
            weight = weightBytes,
            weightByteOffset = (w.storage as Storage.Heap).arrayOffset,
            inputDim = a.shape[1],
            outputDim = w.shape[0],
            out = outFloats,
            outOffset = (out.storage as Storage.Heap).arrayOffset,
        )
    }
}
