package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * The fused multi-plane lm_head gemv a platform pack supplies (#1150) — the seam over the
 * vendored NeoGPU `skainet_ternary_lmhead_stage1` (#1137). Array-shaped like the other ternary
 * seams; the pack does the view unwrapping once.
 *
 * One call computes **planes [firstPlane, firstPlane+4)** of a `BITNET_PLANES` weight with the
 * fused plane weights `{1, ⅓, ⅑, 1/27}` and the FP16 row scales applied:
 *
 *   `out[o] = rowScale[o] · Σ_{q=0}^{3} (1/3^q) · Σ_k in[k] · code(firstPlane+q, o, k)`
 *
 * The full 8-plane matmul is two calls (`firstPlane` 0 and 4) combined as `s0 + s4 / 81` — which
 * is how [NativeTernaryPlanesViewKernel] keeps the dispatch invariant *matmul == decoded matmul*.
 * `inputDim % 4 == 0`; [rowScaleByteOffset] must be 2-byte aligned.
 */
public interface TernaryLmheadNative {
    /** A name for logs and traces, e.g. `ffm`. */
    public val name: String

    public fun lmheadStage1(
        activation: FloatArray,
        activationOffset: Int,
        weight: ByteArray,
        planesByteOffset: Int,
        planeStrideBytes: Int,
        rowScaleByteOffset: Int,
        inputDim: Int,
        outputDim: Int,
        out: FloatArray,
        outOffset: Int,
    )
}

/**
 * Installs the exact FP32 × `BITNET_PLANES` path (#1150). Same contract as
 * [TernaryF32KernelPack]: registration **only with a native kernel** — without it, dispatch falls
 * back to the decoding reference matmul (correct, slow), told through [warn], never a crash.
 */
public object TernaryPlanesKernelPack {

    /** What [install] returns when no native kernel is available and nothing was registered. */
    public const val NOT_INSTALLED: String = "ternary_planes_matmul/not-installed"

    public fun install(
        native: TernaryLmheadNative? = null,
        capabilities: Set<String> = emptySet(),
        warn: (String) -> Unit = {},
    ): String {
        if (native == null) {
            warn(
                "ternary_planes_matmul: no native kernel available — BITNET_PLANES matmuls decode " +
                    "through the reference path. Add the native artifact for the fused LUT path; " +
                    "nothing else changes.",
            )
            return NOT_INSTALLED
        }
        val kernel = NativeTernaryPlanesViewKernel(native, TernaryPlanesMatmulKernel.keyFor().copy(capabilities = capabilities))
        KernelDispatch.register(kernel)
        KernelDispatch.register(NativeTernaryPlanesViewKernel(native, TernaryPlanesMatmulKernel.keyFor()))
        return kernel.name
    }
}

/**
 * The portable reference for FP32 × `BITNET_PLANES` — decodes each weight row (all 8 planes ×
 * row scale) through [TernaryCodec] and accumulates in FP32. The correctness oracle and the
 * in-kernel fallback; deliberately not a dispatch entry on its own.
 */
public class TernaryPlanesMatmulKernel(override val key: KernelKey) : ViewKernel {

    override val name: String get() = "ternary_planes_matmul/reference"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "ternary_planes_matmul takes (activation, weight), got ${inputs.size}" }
        val a = inputs[0]
        val w = inputs[1]
        require(w.format.encoding == TensorEncoding.BITNET_PLANES) {
            "weight must be ${TensorEncoding.BITNET_PLANES}, was ${w.format}"
        }
        require(a.shape.rank == 2 && w.shape.rank == 2 && out.shape.rank == 2) { "ternary_planes_matmul is 2-D" }
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions differ: activation k=$k, weight k=${w.shape[1]}" }
        require(out.shape[0] == rows && out.shape[1] == n) { "out must be [$rows, $n], was ${out.shape}" }
        if (rows == 0 || n == 0) return

        val heap = w.storage as? Storage.Heap
            ?: throw UnsupportedOperationException("ternary_planes_matmul reads weights from heap storage in this milestone")
        val bytes = heap.bytes ?: throw UnsupportedOperationException("BITNET_PLANES weights need byte storage")
        val byteOffset = heap.arrayOffset

        val decodedRow = FloatArray(k)
        for (o in 0 until n) {
            TernaryCodec.decodeBitNetPlanesRow(bytes, n, k, o, decodedRow, 0, byteOffset)
            for (r in 0 until rows) {
                var acc = 0f
                for (i in 0 until k) acc += a.get(r, i) * decodedRow[i]
                out.set(r, o, value = acc)
            }
        }
    }

    public companion object {
        /** The exact key: FP32 dense contiguous × `BITNET_PLANES` row-major. */
        public fun keyFor(): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, TensorEncoding.BITNET_PLANES), LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )
    }
}

/**
 * A [ViewKernel] over a [TernaryLmheadNative]: two fused stage-1 calls per activation row —
 * planes 0–3 and planes 4–7 — combined as `s0 + s4 / 81`, so the result is the **full 8-plane**
 * matmul and the dispatch invariant *matmul == decoded matmul* holds exactly. (Stage-1-only
 * scoring with top-k rescoring is an application-level decision made through the codec's row
 * accessors, never through dispatch.)
 *
 * Falls back to the reference for non-heap storage, strided views, or `k % 4 != 0`.
 */
public class NativeTernaryPlanesViewKernel(
    private val native: TernaryLmheadNative,
    override val key: KernelKey,
) : ViewKernel {

    override val name: String get() = "ternary_planes_matmul/${native.name}"

    private val reference = TernaryPlanesMatmulKernel(key)

    override fun run(inputs: List<TensorView>, out: TensorView) {
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        val activationFloats = (a.storage as? Storage.Heap)?.floats
        val weightBytes = (w.storage as? Storage.Heap)?.bytes
        val outFloats = (out.storage as? Storage.Heap)?.floats
        if (k % 4 != 0 || activationFloats == null || weightBytes == null || outFloats == null ||
            !a.isContiguous || !out.isContiguous
        ) {
            reference.run(inputs, out)
            return
        }
        if (rows == 0 || n == 0) return
        val aOffset = (a.storage as Storage.Heap).arrayOffset
        val wOffset = (w.storage as Storage.Heap).arrayOffset
        val outOffset = (out.storage as Storage.Heap).arrayOffset
        val planeStride = TensorEncoding.BITNET_PLANES.planeStrideBytes(n, k)
        val scalesOffset = TensorEncoding.BITNET_PLANES.rowScalesByteOffset(n, k)
        val high = FloatArray(n)
        for (r in 0 until rows) {
            // planes 0–3, row scales applied by the kernel
            native.lmheadStage1(
                activation = activationFloats, activationOffset = aOffset + r * k,
                weight = weightBytes, planesByteOffset = wOffset,
                planeStrideBytes = planeStride, rowScaleByteOffset = wOffset + scalesOffset,
                inputDim = k, outputDim = n,
                out = outFloats, outOffset = outOffset + r * n,
            )
            // planes 4–7, same fused weights — worth 1/3⁴ of the total
            native.lmheadStage1(
                activation = activationFloats, activationOffset = aOffset + r * k,
                weight = weightBytes, planesByteOffset = wOffset + planeStride * 4,
                planeStrideBytes = planeStride, rowScaleByteOffset = wOffset + scalesOffset,
                inputDim = k, outputDim = n,
                out = high, outOffset = 0,
            )
            val base = outOffset + r * n
            for (o in 0 until n) outFloats[base + o] += high[o] / 81f
        }
    }
}
