package sk.ainet.exec.kernel

import java.nio.file.Files
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertTrue
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.lang.memory.MappedBufferStorage
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.BufferPackedTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * #1191 end-to-end on the JVM: a mapped Q4_K weight dispatches to the FFM row-major kernel
 * (zero-copy over the mmap), produces the same numbers as the decoded reference, and the trace
 * names the kernel that ran. Also pins #1193's visibility contract: when the fast path cannot
 * serve, the fallback announces itself in the trace instead of silently decoding.
 */
class FfmRowMajorDispatchTest {

    @AfterTest
    fun reset() = KernelDispatch.clearForTesting()

    private fun q4kPayload(rows: Int, k: Int, seed: Int): ByteArray {
        val bpr = k / 256
        val rng = Random(seed)
        val bytes = ByteArray(rows * bpr * 144)
        rng.nextBytes(bytes)
        for (b in 0 until rows * bpr) {
            bytes[b * 144] = 0x00; bytes[b * 144 + 1] = 0x3C
            bytes[b * 144 + 2] = 0x00; bytes[b * 144 + 3] = 0x3C
        }
        return bytes
    }

    @Test
    fun mapped_q4k_dispatches_to_the_ffm_row_major_kernel() {
        KernelDispatch.clearForTesting()
        FfmRowMajorKernelPack.install()

        val rows = 64
        val k = 512
        val payload = q4kPayload(rows, k, seed = 6)
        val file = Files.createTempFile("ffm-rm", ".bin")
        try {
            Files.write(file, payload)
            val storage = MappedBufferStorage.map(file, 0, payload.size.toLong())
            val weight = BufferPackedTensorData(Shape(rows, k), storage, TensorEncoding.Q4_K)

            val act = FloatArray(k) { Random(7 + it).nextFloat() - 0.5f }
            val actStorage = Storage.Heap.wrap(act, mutable = false)
            val actView = TensorView.dense(actStorage, Shape(1, k), FP32)
            val out = FloatArray(rows)
            val outView = TensorView.dense(Storage.Heap.wrap(out, mutable = true), Shape(1, rows), FP32)

            val sink = RecordingTraceSink()
            KernelDispatch.matmul(actView, weight.packedView, outView, sink = sink)

            // Trace names the FFM kernel — proof it served, not the reference.
            val runs = sink.events().filterIsInstance<TraceEvent.KernelRun>()
            assertTrue(runs.any { "ffm-rowmajor-Q4_K" in it.kernel }, "expected ffm kernel in trace, got: ${runs.map { it.kernel }}")
            assertTrue(runs.none { "reference-fallback" in it.kernel }, "no fallback expected: ${runs.map { it.kernel }}")

            // Numbers equal the feed-order kernel on permuted bytes — the same W4A8 family
            // (Q4_K quantizes the activation to int8; the exact-float decode is NOT the oracle,
            // see #944), so only -ffast-math reassociation separates them.
            val bpr = k / 256
            val feed = ByteArray(payload.size)
            for (o in 0 until rows) for (b in 0 until bpr) {
                payload.copyInto(feed, (b * rows + o) * 144, (o * bpr + b) * 144, (o * bpr + b + 1) * 144)
            }
            val expected = FloatArray(rows)
            NativeQ4KMatmulKernel.matmul(act, 0, feed, 0, k, rows, expected, 0)
            for (o in 0 until rows) {
                assertTrue(kotlin.math.abs(expected[o] - out[o]) <= maxOf(1e-5f, 1e-5f * kotlin.math.abs(expected[o])),
                    "row $o: dispatch ${out[o]} vs feed-order ${expected[o]}")
            }
        } finally {
            Files.deleteIfExists(file)
        }
    }

    @Test
    fun a_fallback_announces_itself_in_the_trace() {
        // #1193: a packed kernel that cannot serve its operands must say so. Byte-backed output
        // storage is un-servable for every packed matmul bridge — the kernel punts to the
        // decoding reference AND leaves a trace line naming itself and the reason.
        val kernel = sk.ainet.backend.api.kernel.PackedViewMatmulKernel(
            "test", "Q4_K",
            sk.ainet.backend.api.kernel.PackedViewMatmulKernel.keyFor(TensorEncoding.Q4_K),
        ) { _, _, _, _, _, _, _, _ -> }

        val rows = 2
        val k = 256
        val payload = q4kPayload(rows, k, seed = 8)
        val weightView = TensorView.packed(
            storage = Storage.Heap.wrap(payload, mutable = false),
            shape = Shape(rows, k),
            encoding = TensorEncoding.Q4_K,
            decoder = sk.ainet.lang.memory.PackedBlockDecoder(Q4_KBlockTensorData(Shape(rows, k), payload)),
            blockOrder = sk.ainet.lang.memory.BlockOrder.INPUT_BLOCK_MAJOR,
        )
        // A STRIDED activation (every second float of a doubled backing row): the C kernel
        // would mis-index it, so the bridge must punt — and the reference reads any layout.
        val actBacking = FloatArray(k * 2) { Random(9 + it).nextFloat() - 0.5f }
        val actView = TensorView(
            Shape(1, k),
            sk.ainet.lang.memory.Format.dense(FP32),
            sk.ainet.lang.memory.Layout(Shape(1, k), intArrayOf(k * 2, 2)),
            Storage.Heap.wrap(actBacking, mutable = false),
            null,
        )
        val outView = TensorView.dense(Storage.Heap.wrap(FloatArray(rows), mutable = true), Shape(1, rows), FP32)

        val sink = RecordingTraceSink()
        kernel.run(listOf(actView, weightView), outView, sink)

        val runs = sink.events().filterIsInstance<TraceEvent.KernelRun>()
        assertTrue(
            runs.any { "reference-fallback" in it.kernel && "strided" in it.kernel },
            "fallback must announce itself; trace: ${runs.map { it.kernel }}",
        )
    }
}
