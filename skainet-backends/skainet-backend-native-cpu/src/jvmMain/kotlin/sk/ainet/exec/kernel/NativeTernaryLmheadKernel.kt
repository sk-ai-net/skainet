package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.TernaryLmheadNative
import sk.ainet.backend.api.kernel.TernaryPlanesKernelPack

/**
 * Native (FFM) downcall to the vendored NeoGPU fused 4-plane lm_head kernel (#1150).
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_ternary_lmhead_stage1(
 *       const float* input,    int32_t input_offset,
 *       const uint8_t* planes, int32_t planes_byte_offset, int32_t plane_stride_bytes,
 *       const uint16_t* row_scale, int32_t row_scale_offset,
 *       int32_t input_dim, int32_t output_dim,
 *       float* output, int32_t output_offset);
 *
 * One call computes four planes with fused weights {1, ⅓, ⅑, 1/27} and applies the FP16 row
 * scales; the pack's view kernel makes two calls for the full 8-plane result. The row-scale
 * pointer is the weight segment sliced at [TernaryLmheadNative.lmheadStage1]'s
 * `rowScaleByteOffset` (must be 2-byte aligned — it always is, plane strides are `rows·cols/4`
 * with `cols % 4 == 0`). The C side threads internally with pthreads at any output_dim.
 *
 * Copy-in/copy-out per call like every FFM kernel here — fine for benches and small heads;
 * a persistent off-heap weight arena is the follow-up for a 128k-vocab lm_head.
 */
public object NativeTernaryLmheadKernel : TernaryLmheadNative {

    override val name: String get() = "ffm"

    public fun isAvailable(): Boolean = handle != null

    /** Register with [TernaryPlanesKernelPack] when the bundled library resolves. */
    public fun install(warn: (String) -> Unit = {}): String =
        TernaryPlanesKernelPack.install(if (isAvailable()) this else null, warn = warn)

    override fun lmheadStage1(
        activation: FloatArray, activationOffset: Int,
        weight: ByteArray, planesByteOffset: Int,
        planeStrideBytes: Int, rowScaleByteOffset: Int,
        inputDim: Int, outputDim: Int,
        out: FloatArray, outOffset: Int,
    ) {
        require(inputDim % 4 == 0) {
            "NativeTernaryLmheadKernel: inputDim must be a multiple of 4; got $inputDim"
        }
        require(rowScaleByteOffset % 2 == 0) {
            "NativeTernaryLmheadKernel: rowScaleByteOffset must be 2-byte aligned; got $rowScaleByteOffset"
        }
        if (outputDim == 0) return

        val mh = handle
            ?: error("NativeTernaryLmheadKernel invoked while native library unavailable")

        val inputReachFloats = if (inputDim == 0) 0 else activationOffset + inputDim
        val outputReachFloats = outOffset + outputDim

        Arena.ofConfined().use { arena ->
            val fAlign = ValueLayout.JAVA_FLOAT.byteAlignment()
            val inputSeg: MemorySegment = if (inputReachFloats > 0)
                arena.allocate(inputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)
            else MemorySegment.NULL
            val weightSeg: MemorySegment = arena.allocate(weight.size.toLong(), ValueLayout.JAVA_SHORT.byteAlignment())
            val outputSeg: MemorySegment =
                arena.allocate(outputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)

            if (inputReachFloats > 0) {
                MemorySegment.copy(activation, 0, inputSeg, ValueLayout.JAVA_FLOAT, 0L, inputReachFloats)
            }
            MemorySegment.copy(weight, 0, weightSeg, ValueLayout.JAVA_BYTE, 0L, weight.size)
            // Round-trip the output reach so content before outOffset survives copy-back.
            MemorySegment.copy(out, 0, outputSeg, ValueLayout.JAVA_FLOAT, 0L, outputReachFloats)

            mh.invoke(
                inputSeg, activationOffset,
                weightSeg, planesByteOffset, planeStrideBytes,
                weightSeg.asSlice(rowScaleByteOffset.toLong()), 0,
                inputDim, outputDim,
                outputSeg, outOffset,
            )

            MemorySegment.copy(outputSeg, ValueLayout.JAVA_FLOAT, 0L, out, 0, outputReachFloats)
        }
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_ternary_lmhead_stage1").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // input
            ValueLayout.JAVA_INT,   // input_offset
            ValueLayout.ADDRESS,    // planes
            ValueLayout.JAVA_INT,   // planes_byte_offset
            ValueLayout.JAVA_INT,   // plane_stride_bytes
            ValueLayout.ADDRESS,    // row_scale
            ValueLayout.JAVA_INT,   // row_scale_offset
            ValueLayout.JAVA_INT,   // input_dim
            ValueLayout.JAVA_INT,   // output_dim
            ValueLayout.ADDRESS,    // output
            ValueLayout.JAVA_INT,   // output_offset
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
