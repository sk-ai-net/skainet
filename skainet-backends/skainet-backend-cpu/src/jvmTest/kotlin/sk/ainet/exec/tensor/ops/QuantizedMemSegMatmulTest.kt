package sk.ainet.exec.tensor.ops

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.Q4MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q4MemorySegmentTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KTensorData
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q8MemorySegmentTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena

/**
 * Tests for quantized MemorySegment matmul dispatch and transpose.
 */
class QuantizedMemSegMatmulTest {

    private val dataFactory = DenseTensorDataFactory()
    private val ops = DefaultCpuOpsJvm(dataFactory)

    private fun fpTensor(shape: Shape, values: FloatArray): Tensor<FP32, Float> {
        val data = dataFactory.fromFloatArray<FP32, Float>(shape, FP32::class, values)
        return VoidOpsTensor(data, FP32::class)
    }

    @Suppress("UNCHECKED_CAST")
    private fun q4Tensor(shape: Shape, bytes: ByteArray, arena: Arena): Tensor<FP32, Float> {
        val q4 = Q4MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
        return VoidOpsTensor(q4 as TensorData<FP32, Float>, FP32::class)
    }

    @Suppress("UNCHECKED_CAST")
    private fun q8Tensor(shape: Shape, bytes: ByteArray, arena: Arena): Tensor<FP32, Float> {
        val q8 = Q8MemorySegmentTensorData.fromRawBytes(shape, bytes, arena)
        return VoidOpsTensor(q8 as TensorData<FP32, Float>, FP32::class)
    }

    /**
     * Encode a single Q4_0 block: 32 float values -> 18 bytes (2 scale + 16 packed nibbles).
     * Uses the canonical ggml *split* layout: code[j] is the low nibble of
     * byte j, code[j+16] is the high nibble of byte j.
     */
    private fun encodeQ4_0Block(values: FloatArray): ByteArray {
        require(values.size == 32)
        val maxAbs = values.maxOf { abs(it) }
        val scale = if (maxAbs == 0f) 0f else maxAbs / 7f
        val scaleHalf = floatToHalf(scale)

        val codes = IntArray(32) { i ->
            if (scale == 0f) 8
            else (values[i] / scale + 8).toInt().coerceIn(0, 15)
        }

        val out = ByteArray(18)
        out[0] = (scaleHalf and 0xFF).toByte()
        out[1] = ((scaleHalf shr 8) and 0xFF).toByte()
        for (j in 0 until 16) {
            out[2 + j] = ((codes[j + 16] shl 4) or codes[j]).toByte()
        }
        return out
    }

    /**
     * Encode a single Q8_0 block: 32 float values -> 34 bytes (2 scale + 32 int8 codes).
     */
    private fun encodeQ8_0Block(values: FloatArray): ByteArray {
        require(values.size == 32)
        val maxAbs = values.maxOf { abs(it) }
        val scale = if (maxAbs == 0f) 0f else maxAbs / 127f
        val scaleHalf = floatToHalf(scale)

        val out = ByteArray(34)
        out[0] = (scaleHalf and 0xFF).toByte()
        out[1] = ((scaleHalf shr 8) and 0xFF).toByte()
        for (i in 0 until 32) {
            val code = if (scale == 0f) 0 else (values[i] / scale).toInt().coerceIn(-128, 127)
            out[2 + i] = code.toByte()
        }
        return out
    }

    private fun floatToHalf(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exp = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (exp <= 0) return sign
        if (exp >= 31) return sign or 0x7C00
        return sign or (exp shl 10) or (mant ushr 13)
    }

    // ── Transpose Tests ─────────────────────────────────────────────────────

    @Test
    fun `Q4 lazy transpose swaps shape dimensions`() {
        val arena = Arena.ofConfined()
        val block1 = encodeQ4_0Block(FloatArray(32) { 0.1f * it })
        val block2 = encodeQ4_0Block(FloatArray(32) { -0.1f * it })
        val bytes = block1 + block2
        val t = q4Tensor(Shape(2, 32), bytes, arena)

        val transposed = ops.transpose(t)
        assertEquals(Shape(32, 2), transposed.shape)
        assertTrue(transposed.data is Q4MemorySegmentMarker,
            "Transpose should preserve Q4 MemorySegment data type")
        arena.close()
    }

    @Test
    fun `Q8 lazy transpose swaps shape dimensions`() {
        val arena = Arena.ofConfined()
        val block1 = encodeQ8_0Block(FloatArray(32) { 0.05f * it })
        val block2 = encodeQ8_0Block(FloatArray(32) { -0.05f * it })
        val bytes = block1 + block2
        val t = q8Tensor(Shape(2, 32), bytes, arena)

        val transposed = ops.transpose(t)
        assertEquals(Shape(32, 2), transposed.shape)
        assertTrue(transposed.data is Q8MemorySegmentMarker,
            "Transpose should preserve Q8 MemorySegment data type")
        arena.close()
    }

    @Test
    fun `Q4_K lazy transpose swaps shape dimensions and keeps packed bytes`() {
        // Two Q4_K blocks (256 elements each = 144 bytes each). Content doesn't
        // matter for the shape/identity invariant — use random bytes.
        val numBlocks = 2
        val bytes = ByteArray(numBlocks * Q4_KTensorData.BYTES_PER_BLOCK) { i -> (i and 0x7F).toByte() }
        val q4k = Q4_KBlockTensorData(
            Shape(numBlocks, Q4_KTensorData.BLOCK_SIZE),
            bytes
        )
        @Suppress("UNCHECKED_CAST")
        val tensor: Tensor<FP32, Float> = VoidOpsTensor(q4k as TensorData<FP32, Float>, FP32::class)

        val transposed = ops.relayoutPackedWeightForKernels(tensor)
        assertEquals(Shape(Q4_KTensorData.BLOCK_SIZE, numBlocks), transposed.shape)
        assertTrue(
            transposed.data is Q4_KTensorData,
            "transpose must preserve Q4_K packed layout, got ${transposed.data::class.simpleName}"
        )
        // `cols == BLOCK_SIZE` means exactly one block per row (blocksPerInputDim ==
        // 1), so the physical block-grid transpose (see `DefaultCpuOpsBase
        // .transposePackedBlocks`) is a content-preserving permutation here — but it
        // is still a genuine copy (a new array), not the old bare shape-swap-only
        // "zero-copy" behaviour, which was only byte-order-correct in this exact
        // single-block-per-row case and silently wrong for every wider row
        // (SKaiNET-transformers#307). Assert content equality, not reference
        // identity — the specialized branch (vs. the fallback per-element
        // transpose, which would crash on Byte → Float casts, the original
        // regression this test guarded) is still exercised either way.
        val transposedPacked = (transposed.data as Q4_KTensorData).packedData
        assertTrue(
            transposedPacked.contentEquals(bytes),
            "Q4_K transpose must preserve block content for a single-block-per-row weight",
        )
    }

    @Test
    fun `Q6_K lazy transpose swaps shape dimensions and keeps packed bytes`() {
        val numBlocks = 2
        val bytes = ByteArray(numBlocks * Q6_KTensorData.BYTES_PER_BLOCK) { i -> (i and 0x7F).toByte() }
        val q6k = Q6_KBlockTensorData(
            Shape(numBlocks, Q6_KTensorData.BLOCK_SIZE),
            bytes
        )
        @Suppress("UNCHECKED_CAST")
        val tensor: Tensor<FP32, Float> = VoidOpsTensor(q6k as TensorData<FP32, Float>, FP32::class)

        val transposed = ops.relayoutPackedWeightForKernels(tensor)
        assertEquals(Shape(Q6_KTensorData.BLOCK_SIZE, numBlocks), transposed.shape)
        assertTrue(
            transposed.data is Q6_KTensorData,
            "transpose must preserve Q6_K packed layout, got ${transposed.data::class.simpleName}"
        )
        // See the Q4_K case above: single block per row (cols == BLOCK_SIZE) makes
        // the physical block-grid transpose content-preserving but still a copy.
        val transposedPacked = (transposed.data as Q6_KTensorData).packedData
        assertTrue(
            transposedPacked.contentEquals(bytes),
            "Q6_K transpose must preserve block content for a single-block-per-row weight",
        )
    }

    // ── Q4_0 Matmul Tests ───────────────────────────────────────────────────

    @Test
    fun `Q4 MemorySegment matmul produces correct output shape`() {
        val arena = Arena.ofConfined()
        val inputDim = 32
        val outputDim = 2

        val weightBytes = ByteArray(0).let {
            var buf = ByteArray(0)
            repeat(outputDim) { buf += encodeQ4_0Block(FloatArray(inputDim) { 0.1f }) }
            buf
        }
        val weight = q4Tensor(Shape(outputDim, inputDim), weightBytes, arena)
        val input = fpTensor(Shape(1, inputDim), FloatArray(inputDim) { 1f })

        val transposedWeight = ops.transpose(weight)
        assertEquals(Shape(inputDim, outputDim), transposedWeight.shape)

        val result = ops.matmul(input, transposedWeight)
        assertEquals(Shape(1, outputDim), result.shape)
        arena.close()
    }

    @Test
    fun `Q4 MemorySegment matmul matches dequantized reference`() {
        val arena = Arena.ofConfined()
        val inputDim = 32
        val outputDim = 2

        val weightValues = Array(outputDim) { row ->
            FloatArray(inputDim) { col -> 0.3f * (row + 1) * ((col % 5) - 2) / 5f }
        }
        var weightBytes = ByteArray(0)
        for (row in weightValues) weightBytes += encodeQ4_0Block(row)
        val weight = q4Tensor(Shape(outputDim, inputDim), weightBytes, arena)

        val dequantized = (weight.data as Q4MemorySegmentTensorData).copyToFloatArray()

        val inputValues = FloatArray(inputDim) { (it + 1).toFloat() / inputDim }
        val input = fpTensor(Shape(1, inputDim), inputValues)

        // Reference: manual dot product against dequantized weights
        val expected = FloatArray(outputDim) { row ->
            var sum = 0f
            for (k in 0 until inputDim) sum += inputValues[k] * dequantized[row * inputDim + k]
            sum
        }

        val result = ops.matmulWeightTransposed(input, weight)
        val resultData = result.data.copyToFloatArray()

        assertEquals(outputDim, resultData.size)
        for (i in 0 until outputDim) {
            assertEquals(expected[i], resultData[i], 0.5f,
                "Q4 matmul mismatch at output $i: expected=${expected[i]}, actual=${resultData[i]}")
        }
        arena.close()
    }

    // ── Q8_0 Matmul Tests ───────────────────────────────────────────────────

    @Test
    fun `Q8 MemorySegment matmul produces correct output shape`() {
        val arena = Arena.ofConfined()
        val inputDim = 32
        val outputDim = 2

        var weightBytes = ByteArray(0)
        repeat(outputDim) { weightBytes += encodeQ8_0Block(FloatArray(inputDim) { 0.1f }) }
        val weight = q8Tensor(Shape(outputDim, inputDim), weightBytes, arena)
        val input = fpTensor(Shape(1, inputDim), FloatArray(inputDim) { 1f })

        val transposedWeight = ops.transpose(weight)
        val result = ops.matmul(input, transposedWeight)
        assertEquals(Shape(1, outputDim), result.shape)
        arena.close()
    }

    @Test
    fun `Q8 MemorySegment matmul matches dequantized reference`() {
        val arena = Arena.ofConfined()
        val inputDim = 32
        val outputDim = 2

        val weightValues = Array(outputDim) { row ->
            FloatArray(inputDim) { col -> 0.3f * (row + 1) * ((col % 5) - 2) / 5f }
        }
        var weightBytes = ByteArray(0)
        for (row in weightValues) weightBytes += encodeQ8_0Block(row)
        val weight = q8Tensor(Shape(outputDim, inputDim), weightBytes, arena)

        val dequantized = (weight.data as Q8MemorySegmentTensorData).copyToFloatArray()

        val inputValues = FloatArray(inputDim) { (it + 1).toFloat() / inputDim }
        val input = fpTensor(Shape(1, inputDim), inputValues)

        val expected = FloatArray(outputDim) { row ->
            var sum = 0f
            for (k in 0 until inputDim) sum += inputValues[k] * dequantized[row * inputDim + k]
            sum
        }

        val result = ops.matmulWeightTransposed(input, weight)
        val resultData = result.data.copyToFloatArray()

        assertEquals(outputDim, resultData.size)
        for (i in 0 until outputDim) {
            assertEquals(expected[i], resultData[i], 0.1f,
                "Q8 matmul mismatch at output $i: expected=${expected[i]}, actual=${resultData[i]}")
        }
        arena.close()
    }

    @Test
    fun `Q8 matmul with a slab-backed scoped activation matches the dense-activation result`() {
        val arena = Arena.ofConfined()
        val inputDim = 32
        val outputDim = 2

        val weightValues = Array(outputDim) { row ->
            FloatArray(inputDim) { col -> 0.3f * (row + 1) * ((col % 5) - 2) / 5f }
        }
        var weightBytes = ByteArray(0)
        for (row in weightValues) weightBytes += encodeQ8_0Block(row)
        val weight = q8Tensor(Shape(outputDim, inputDim), weightBytes, arena)

        val inputValues = FloatArray(inputDim) { (it + 1).toFloat() / inputDim }
        val expected = ops.matmulWeightTransposed(fpTensor(Shape(1, inputDim), inputValues), weight)
            .data.copyToFloatArray()

        // The forward-scope shape of the same activation: a StorageFloatTensorData over a slab
        // slice at a NONZERO offset — what a ScopedExecutionContext hands every op mid-step
        // (#1145/#1146). Before chooseQuantizedMatmul2D grew its dense-FP32 fallback this fell
        // through to matmulGeneric, whose per-element get() on the Q8 weight returns raw codes —
        // silently wrong results, not an error.
        sk.ainet.lang.memory.ForwardScope(1024).use { scope ->
            scope.allocateFloats(7) // push the next allocation off offset 0
            val st = scope.allocateFloats(inputDim)
            inputValues.copyInto(st.floats!!, st.arrayOffset, 0, inputDim)
            @Suppress("UNCHECKED_CAST")
            val slabData = sk.ainet.lang.tensor.data.StorageFloatTensorData<FP32>(Shape(1, inputDim), st)
            val scoped: Tensor<FP32, Float> = VoidOpsTensor(slabData as TensorData<FP32, Float>, FP32::class)
            val actual = ops.matmulWeightTransposed(scoped, weight).data.copyToFloatArray()

            assertEquals(expected.size, actual.size)
            for (i in expected.indices) {
                assertEquals(
                    expected[i], actual[i], 0f,
                    "scoped-activation mismatch at $i: dense=${expected[i]} scoped=${actual[i]}"
                )
            }
        }
        arena.close()
    }

    // ── Batched Matmul Test ─────────────────────────────────────────────────

    @Test
    fun `Q4 batched matmul produces correct output shape`() {
        val arena = Arena.ofConfined()
        val batchSize = 3
        val inputDim = 32
        val outputDim = 2

        var weightBytes = ByteArray(0)
        repeat(outputDim) { weightBytes += encodeQ4_0Block(FloatArray(inputDim) { 0.1f }) }
        val weight = q4Tensor(Shape(outputDim, inputDim), weightBytes, arena)
        val input = fpTensor(Shape(batchSize, inputDim), FloatArray(batchSize * inputDim) { 1f })

        val result = ops.matmulWeightTransposed(input, weight)
        assertEquals(Shape(batchSize, outputDim), result.shape)
        arena.close()
    }

    // ── Q6_K + MemorySegment-backed activation (SKaiNET#991) ──────────────────

    /**
     * Regression test for SKaiNET#991. Real attention-layer activations produced
     * by [DirectCpuExecutionContext] wired with [MemorySegmentTensorDataFactory]
     * (the config every production caller uses — see KLlamaJava.loadGGUF in
     * SKaiNET-transformers) are `MemorySegmentTensorData`, not
     * `FloatArrayTensorData`. `chooseQuantizedMatmul` (this class) intentionally
     * does not intercept Q6_K/Q5_1/Q5_0 — the comment above its `when(bData)`
     * block says they're "handled in DefaultCpuOpsBase via the kernel registry" —
     * but `DefaultCpuOpsBase.chooseQuantizedMatmulHeap` required
     * `a.data as? FloatArrayTensorData<*>` and silently returned null for
     * anything else, so those quant types fell all the way through to
     * `matmulGeneric`, which has no packed-quant handling and threw
     * `ClassCastException: class java.lang.Byte cannot be cast to class
     * java.lang.Float` reading the raw packed bytes as if they were `Float`.
     *
     * Fixed by having `chooseQuantizedMatmulHeap` call the universal
     * `TensorData.copyToFloatArray()` instead of requiring the
     * `FloatArrayTensorData` subtype specifically.
     */
    @Test
    fun `Q6_K matmul with MemorySegment-backed FP32 activation does not throw and stays finite`() {
        val inputDim = Q6_KTensorData.BLOCK_SIZE // exactly one block per row
        val outputDim = 2
        val numBlocks = outputDim

        val weightBytes = ByteArray(numBlocks * Q6_KTensorData.BYTES_PER_BLOCK) { i -> (i and 0x3F).toByte() }
        // Force a small, finite half-float scale per block (last 2 bytes of each
        // 210-byte Q6_K block) so we don't synthesize a NaN/Inf scale — mirrors
        // the safeguard in Q6KMatmulTest.randomQ6KBytes. 0x3C00 = 1.0f16.
        for (block in 0 until numBlocks) {
            val dOffset = block * Q6_KTensorData.BYTES_PER_BLOCK + 208
            weightBytes[dOffset] = 0x00.toByte()
            weightBytes[dOffset + 1] = 0x3C.toByte()
        }
        @Suppress("UNCHECKED_CAST")
        val weight: Tensor<FP32, Float> = VoidOpsTensor(
            Q6_KBlockTensorData(Shape(numBlocks, inputDim), weightBytes) as TensorData<FP32, Float>,
            FP32::class,
        )

        // MemorySegmentTensorDataFactory, not DenseTensorDataFactory — this is the
        // one detail that reproduces the real bug. `fpTensor()` above (used by
        // every other test in this file) goes through DenseTensorDataFactory and
        // yields FloatArrayTensorData, which never exercised the broken path.
        val memSegFactory = MemorySegmentTensorDataFactory()
        val inputData = memSegFactory.fromFloatArray<FP32, Float>(
            Shape(1, inputDim), FP32::class, FloatArray(inputDim) { (it + 1).toFloat() / inputDim },
        )
        val input: Tensor<FP32, Float> = VoidOpsTensor(inputData, FP32::class)

        val transposedWeight = ops.relayoutPackedWeightForKernels(weight)
        assertTrue(transposedWeight.data is Q6_KTensorData, "the relayout must preserve Q6_K packed layout")

        val result = ops.matmul(input, transposedWeight)

        assertEquals(Shape(1, outputDim), result.shape)
        for (v in result.data.copyToFloatArray()) {
            assertTrue(v.isFinite(), "Q6_K matmul with MemorySegment-backed input produced a non-finite value: $v")
        }
    }

    // ── Q4_K + rank-1 (single-token decode) activation ─────────────────────────

    /**
     * Regression test for the "Byte cannot be cast to Float" crash reported against a real
     * EdgeTranslator run: `chooseQuantizedMatmul`/`chooseQuantizedMatmulHeap` both required
     * `a.shape.rank >= 2`, returning null for a rank-1 activation and sending it straight to
     * `matmulGeneric`'s untyped per-element `TensorData.get()`, which — for a packed-quant weight
     * (in production, one wrapped by SKaiNET-transformers' `PreTransposedQ4_K` marker; a plain
     * `Q4_KBlockTensorData` reproduces the same dispatch gap here) — returns the raw packed byte,
     * not a dequantized Float. Real attention forward passes run FP32 batched (rank >= 2) during
     * prefill but drop to a bare `[in]` hidden-state vector once the KV cache is warm and decoding
     * proceeds one token at a time, which is exactly the shape this test exercises.
     */
    @Test
    fun `Q4_K matmul with rank-1 activation does not throw and stays finite`() {
        val inputDim = Q4_KTensorData.BLOCK_SIZE // exactly one block per row
        val outputDim = 2
        val numBlocks = outputDim

        val weightBytes = ByteArray(numBlocks * Q4_KTensorData.BYTES_PER_BLOCK) { i -> (i and 0x3F).toByte() }
        @Suppress("UNCHECKED_CAST")
        val weight: Tensor<FP32, Float> = VoidOpsTensor(
            Q4_KBlockTensorData(Shape(numBlocks, inputDim), weightBytes) as TensorData<FP32, Float>,
            FP32::class,
        )
        val transposedWeight = ops.relayoutPackedWeightForKernels(weight)
        assertTrue(transposedWeight.data is Q4_KTensorData, "the relayout must preserve Q4_K packed layout")

        // Rank 1, not rank 2 — a single-token hidden-state vector, not a `[1, in]` batch.
        val input = fpTensor(Shape(inputDim), FloatArray(inputDim) { (it + 1).toFloat() / inputDim })

        val result = ops.matmul(input, transposedWeight)

        assertEquals(Shape(outputDim), result.shape)
        for (v in result.data.copyToFloatArray()) {
            assertTrue(v.isFinite(), "Q4_K matmul with a rank-1 activation produced a non-finite value: $v")
        }
    }
}
