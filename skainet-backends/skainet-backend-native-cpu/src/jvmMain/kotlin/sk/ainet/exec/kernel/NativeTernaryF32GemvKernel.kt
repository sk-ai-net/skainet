package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.TernaryF32GemvNative
import sk.ainet.backend.api.kernel.TernaryF32KernelPack
import sk.ainet.lang.memory.SegmentStorage
import sk.ainet.lang.memory.Storage

/**
 * Native (FFM) downcall to the vendored NeoGPU ternary LUT kernel.
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_ternary_f32_gemv(
 *       const float* input,    int32_t input_offset,
 *       const uint8_t* weight, int32_t weight_byte_offset,
 *       int32_t input_dim,     int32_t output_dim,
 *       float* output,         int32_t output_offset);
 *
 * Exact FP32 activations against sequentially-packed ternary weights (the
 * `BITNET_B1_58` payload: 4 codes per byte, low bit-pair first,
 * code {0,1,2} → {-1,0,+1}; byte code 3 decodes to +2 — loaders reject it
 * at import). NO scale is applied — the caller owns the per-tensor scale.
 * Unlike the int8 `bitnet_gemv` path there is no activation quantization,
 * so results are exact.
 *
 * The C side needs only baseline NEON (no dotprod) and threads internally
 * with pthreads once outputDim >= 512. Goldens + threading parity are
 * pinned by `NativeTernaryF32GemvKernelTest`.
 *
 * Refs SKaiNET issue #1137 (vendored from anjaustin/neogpu, MIT — see
 * native/src/vendor/neogpu/README.md); implements the [TernaryF32GemvNative]
 * seam so [install] can hand it to `TernaryF32KernelPack` (#1138) — the
 * first ternary FFM consumer.
 */
public object NativeTernaryF32GemvKernel : TernaryF32GemvNative {

    override val name: String get() = "ffm"

    public fun isAvailable(): Boolean = handle != null

    /**
     * Register this kernel with [TernaryF32KernelPack] when the bundled
     * library resolves; without it the pack warns and dispatch keeps the
     * int8-requantize path. Returns the serving kernel name.
     */
    public fun install(warn: (String) -> Unit = {}): String =
        TernaryF32KernelPack.install(if (isAvailable()) this else null, warn = warn)

    override fun gemvPacked(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % 4 == 0) {
            "NativeTernaryF32GemvKernel: inputDim must be a multiple of 4; got $inputDim"
        }
        if (outputDim == 0) return

        val mh = handle
            ?: error("NativeTernaryF32GemvKernel.gemv invoked while native library unavailable")

        // Reach calculations: copy offset + payload into off-heap arenas,
        // same convention as the other Native*MatmulKernel objects.
        val rowBytes = inputDim / 4
        val inputReachFloats = if (inputDim == 0) 0 else inputOffset + inputDim
        val weightReachBytes = if (inputDim == 0 || outputDim == 0) 0
                               else weightByteOffset + rowBytes * outputDim
        val outputReachFloats = outputOffset + outputDim

        Arena.ofConfined().use { arena ->
            val fAlign = ValueLayout.JAVA_FLOAT.byteAlignment()
            val bAlign = ValueLayout.JAVA_BYTE.byteAlignment()

            val inputSeg: MemorySegment = if (inputReachFloats > 0)
                arena.allocate(inputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)
            else MemorySegment.NULL
            val weightSeg: MemorySegment = if (weightReachBytes > 0)
                arena.allocate(weightReachBytes.toLong(), bAlign)
            else MemorySegment.NULL
            val outputSeg: MemorySegment =
                arena.allocate(outputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)

            if (inputReachFloats > 0) {
                MemorySegment.copy(input, 0, inputSeg, ValueLayout.JAVA_FLOAT, 0L, inputReachFloats)
            }
            if (weightReachBytes > 0) {
                MemorySegment.copy(weight, 0, weightSeg, ValueLayout.JAVA_BYTE, 0L, weightReachBytes)
            }
            // Round-trip the existing output reach so array content before
            // outputOffset survives the copy-back (callers loop rows into
            // one array at increasing offsets — a zeroed segment would
            // clobber the rows already written).
            MemorySegment.copy(output, 0, outputSeg, ValueLayout.JAVA_FLOAT, 0L, outputReachFloats)

            mh.invoke(
                inputSeg, inputOffset,
                weightSeg, weightByteOffset,
                inputDim, outputDim,
                outputSeg, outputOffset,
            )

            MemorySegment.copy(outputSeg, ValueLayout.JAVA_FLOAT, 0L, output, 0, outputReachFloats)
        }
    }

    /**
     * The zero-copy face of [gemvPacked] (#1202): when [weight] is a [SegmentStorage] its
     * `MemorySegment` is handed to the native call directly — the weight is never copied, not even
     * once, unlike [gemvPacked] which always stages the weight into a fresh confined arena. This is
     * what makes an off-heap-resident ternary weight (#1202's fix for the ART heap cap) actually
     * fast on the row loop [sk.ainet.backend.api.kernel.NativeTernaryF32ViewKernel] runs, rather
     * than falling back to re-copying the whole matrix on every row.
     *
     * Activation/output still stage through a small confined arena — they're `k`/`n` floats, not
     * the weight matrix, so the cost is the same as [gemvPacked] pays today for those two.
     */
    override fun gemvPackedStorage(
        activation: FloatArray, activationOffset: Int,
        weight: Storage, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        val weightSeg = (weight as? SegmentStorage)?.segment()
        if (weightSeg == null) {
            super.gemvPackedStorage(activation, activationOffset, weight, weightByteOffset, inputDim, outputDim, output, outputOffset)
            return
        }
        require(inputDim % 4 == 0) {
            "NativeTernaryF32GemvKernel: inputDim must be a multiple of 4; got $inputDim"
        }
        if (outputDim == 0) return

        val mh = handle
            ?: error("NativeTernaryF32GemvKernel.gemv invoked while native library unavailable")

        val inputReachFloats = if (inputDim == 0) 0 else activationOffset + inputDim
        val outputReachFloats = outputOffset + outputDim

        Arena.ofConfined().use { arena ->
            val fAlign = ValueLayout.JAVA_FLOAT.byteAlignment()

            val inputSeg: MemorySegment = if (inputReachFloats > 0)
                arena.allocate(inputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)
            else MemorySegment.NULL
            val outputSeg: MemorySegment =
                arena.allocate(outputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)

            if (inputReachFloats > 0) {
                MemorySegment.copy(activation, 0, inputSeg, ValueLayout.JAVA_FLOAT, 0L, inputReachFloats)
            }
            MemorySegment.copy(output, 0, outputSeg, ValueLayout.JAVA_FLOAT, 0L, outputReachFloats)

            mh.invoke(
                inputSeg, activationOffset,
                weightSeg, weightByteOffset,
                inputDim, outputDim,
                outputSeg, outputOffset,
            )

            MemorySegment.copy(outputSeg, ValueLayout.JAVA_FLOAT, 0L, output, 0, outputReachFloats)
        }
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_ternary_f32_gemv").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // input
            ValueLayout.JAVA_INT,   // input_offset
            ValueLayout.ADDRESS,    // weight
            ValueLayout.JAVA_INT,   // weight_byte_offset
            ValueLayout.JAVA_INT,   // input_dim
            ValueLayout.JAVA_INT,   // output_dim
            ValueLayout.ADDRESS,    // output
            ValueLayout.JAVA_INT,   // output_offset
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
