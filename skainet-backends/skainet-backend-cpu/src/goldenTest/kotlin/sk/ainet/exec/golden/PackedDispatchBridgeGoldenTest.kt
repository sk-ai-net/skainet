package sk.ainet.exec.golden

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.backend.api.kernel.PackedViewMatmulKernel
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.exec.golden.GoldenSupport.Packed
import sk.ainet.exec.kernel.ScalarKernelProvider
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.PackedBlockDecoder
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1095 (#973.2): the packed SPI kernels reached through the registry produce **exactly** what
 * calling them directly produces.
 *
 * This is the parity that #1029 could not assert, because nothing said which block order a view's
 * bytes were in. The dispatcher now relayouts a canonical weight into kernel order and selects the
 * packed kernel; the result must be bit-identical to feeding that kernel block-major bytes by hand,
 * which is what `ScalarKernelGoldenTest` already pins. Every weight here is **three blocks wide**,
 * the case where the two orders differ.
 */
class PackedDispatchBridgeGoldenTest {

    private companion object {
        const val OUT = 4
        const val BLOCKS = 3
        const val SEED = 0x5EED_0002L      // the seed ScalarKernelGoldenTest uses
    }

    @BeforeTest fun setUp() {
        KernelDispatch.clearForTesting()
        KernelPacks.installReference()
        KernelPacks.installPacked(ScalarKernelProvider)
    }

    @AfterTest fun tearDown() = KernelDispatch.clearForTesting()

    private fun encodingOf(p: Packed): TensorEncoding = when (p) {
        Packed.Q4_0 -> TensorEncoding.Q4_0
        Packed.Q5_0 -> TensorEncoding.Q5_0
        Packed.Q5_1 -> TensorEncoding.Q5_1
        Packed.Q8_0 -> TensorEncoding.Q8_0
        Packed.Q4_K -> TensorEncoding.Q4_K
        Packed.Q5_K -> TensorEncoding.Q5_K
        Packed.Q6_K -> TensorEncoding.Q6_K
    }

    private fun dataOf(p: Packed, shape: Shape, bytes: ByteArray): PackedBlockStorage = when (p) {
        Packed.Q4_0 -> Q4_0BlockTensorData(shape, bytes)
        Packed.Q5_0 -> Q5_0BlockTensorData(shape, bytes)
        Packed.Q5_1 -> Q5_1BlockTensorData(shape, bytes)
        Packed.Q8_0 -> Q8_0BlockTensorData(shape, bytes)
        Packed.Q4_K -> Q4_KBlockTensorData(shape, bytes)
        Packed.Q5_K -> Q5_KBlockTensorData(shape, bytes)
        Packed.Q6_K -> Q6_KBlockTensorData(shape, bytes)
    }

    private fun bridge(p: Packed) {
        val blocks = GoldenSupport.weightBlocks(p, OUT, BLOCKS, SEED)
        val inputDim = BLOCKS * p.blockSize
        val shape = Shape(OUT, inputDim)
        val input = GoldenSupport.floats(inputDim, SEED + 100)

        // what the kernel is fed by hand today: block-major bytes
        val direct = FloatArray(OUT)
        val spi = requireNotNull(spiKernel(p)) { "${p.name}: the scalar provider must offer this kernel" }
        spi(input, 0, GoldenSupport.blockMajor(blocks), 0, inputDim, OUT, direct, 0)

        // what the registry produces from a canonical view of the same weight
        val canonicalBytes = GoldenSupport.rowMajor(blocks)
        val weight = TensorView.packed(
            Storage.Heap.wrap(canonicalBytes, mutable = false), shape, encodingOf(p),
            PackedBlockDecoder(dataOf(p, shape, canonicalBytes)),
        )
        val activation = TensorView.dense(Storage.Heap.wrap(input), Shape(1, inputDim), FP32)
        val out = TensorView.dense(Storage.Heap.floats(OUT), Shape(1, OUT), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(activation, weight, out, Scope.Ambient, sink, prepackWeights = true)

        val kernelRun = sink.eventsOf<TraceEvent.KernelRun>().single()
        assertTrue(
            kernelRun.kernel.endsWith(p.name),
            "${p.name}: the registry should select the packed kernel, ran '${kernelRun.kernel}'",
        )
        val adapter = sink.eventsOf<TraceEvent.AdapterInserted>().single()
        assertEquals("prepack-input_block_major", adapter.kind, "${p.name}: the relayout must be visible")

        val fromRegistry = FloatArray(OUT) { out.get(0, it) }
        assertContentEquals(direct, fromRegistry, "${p.name}: registry vs direct call must be bit-identical")
        GoldenSupport.check("dispatch-bridge/${p.name}", GoldenSupport.digest(fromRegistry))
    }

    private fun spiKernel(p: Packed): ((FloatArray, Int, ByteArray, Int, Int, Int, FloatArray, Int) -> Unit)? = when (p) {
        Packed.Q4_0 -> ScalarKernelProvider.matmulQ4_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q5_0 -> ScalarKernelProvider.matmulQ5_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q5_1 -> ScalarKernelProvider.matmulQ5_1()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q8_0 -> ScalarKernelProvider.matmulQ8_0()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q4_K -> ScalarKernelProvider.matmulQ4K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q5_K -> ScalarKernelProvider.matmulQ5K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
        Packed.Q6_K -> ScalarKernelProvider.matmulQ6K()?.let { k -> { i, io, w, wo, id, od, o, oo -> k.matmul(i, io, w, wo, id, od, o, oo) } }
    }

    @Test fun q4_0() = bridge(Packed.Q4_0)
    @Test fun q5_0() = bridge(Packed.Q5_0)
    @Test fun q5_1() = bridge(Packed.Q5_1)
    @Test fun q8_0() = bridge(Packed.Q8_0)
    @Test fun q4_K() = bridge(Packed.Q4_K)
    @Test fun q5_K() = bridge(Packed.Q5_K)
    @Test fun q6_K() = bridge(Packed.Q6_K)

    @Test
    fun aWeightAlreadyInKernelOrderCostsNoAdapter() {
        // the shape a caller should hand over in a hot loop: prepacked once, at load (#1097)
        val p = Packed.Q8_0
        val blocks = GoldenSupport.weightBlocks(p, OUT, BLOCKS, SEED)
        val inputDim = BLOCKS * p.blockSize
        val shape = Shape(OUT, inputDim)
        val canonicalBytes = GoldenSupport.rowMajor(blocks)
        val weight = TensorView.packed(
            Storage.Heap.wrap(canonicalBytes, mutable = false), shape, encodingOf(p),
            PackedBlockDecoder(dataOf(p, shape, canonicalBytes)),
        ).prepack(BlockOrder.INPUT_BLOCK_MAJOR)

        val input = GoldenSupport.floats(inputDim, SEED + 100)
        val activation = TensorView.dense(Storage.Heap.wrap(input), Shape(1, inputDim), FP32)
        val out = TensorView.dense(Storage.Heap.floats(OUT), Shape(1, OUT), FP32)
        val sink = RecordingTraceSink()
        KernelDispatch.matmul(activation, weight, out, Scope.Ambient, sink)

        assertTrue(sink.eventsOf<TraceEvent.AdapterInserted>().isEmpty(), "nothing to relayout: it is already in kernel order")
        val direct = FloatArray(OUT)
        spiKernel(p)!!(input, 0, GoldenSupport.blockMajor(blocks), 0, inputDim, OUT, direct, 0)
        assertContentEquals(direct, FloatArray(OUT) { out.get(0, it) })
    }
}
