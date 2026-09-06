package sk.ainet.exec.golden

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.golden.GoldenSupport.Packed
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1034: a packed transpose is metadata.
 *
 * `relayoutPackedWeightForKernels` permutes the block grid byte by byte, because the packed matmul
 * kernels read their weight as input-block-major regardless of its declared shape — the contract
 * that made #968/#971 read garbage from a bare shape swap, and that #973 exists to write down.
 * A `TensorView` needs no such permutation: transposing swaps two strides and moves the block axis
 * ([sk.ainet.lang.memory.Layout.blockAxis]) with it.
 *
 * Both must describe the *same matrix*. This asserts exactly that, for every packed encoding, on
 * JVM and Kotlin/Native alike — and records the transposed values as goldens.
 */
class PackedTransposeGoldenTest {

    private companion object {
        const val ROWS = 4
        const val BLOCKS_PER_ROW = 3
        const val SEED = 0x5EED_0003L
    }

    private val ctx = DirectCpuExecutionContext()

    private fun build(p: Packed, shape: Shape, bytes: ByteArray): PackedBlockStorage = when (p) {
        Packed.Q4_0 -> Q4_0BlockTensorData(shape, bytes)
        Packed.Q5_0 -> Q5_0BlockTensorData(shape, bytes)
        Packed.Q5_1 -> Q5_1BlockTensorData(shape, bytes)
        Packed.Q8_0 -> Q8_0BlockTensorData(shape, bytes)
        Packed.Q4_K -> Q4_KBlockTensorData(shape, bytes)
        Packed.Q5_K -> Q5_KBlockTensorData(shape, bytes)
        Packed.Q6_K -> Q6_KBlockTensorData(shape, bytes)
    }

    @Suppress("UNCHECKED_CAST")
    private fun transposeAgrees(p: Packed) {
        val shape = Shape(ROWS, BLOCKS_PER_ROW * p.blockSize)
        val bytes = GoldenSupport.rowMajor(GoldenSupport.weightBlocks(p, ROWS, BLOCKS_PER_ROW, SEED))
        val data = build(p, shape, bytes)
        val tensor: Tensor<FP32, Float> = ctx.fromData(data as TensorData<FP32, Float>, FP32::class)

        val view = data.packedView
        val transposed = view.transpose()
        assertEquals(Shape(shape[1], shape[0]), transposed.shape, "${p.name}: shape")
        assertEquals(view.storage.id, transposed.storage.id, "${p.name}: the transpose must not touch the bytes")

        // 1. the view transposes what it decodes
        for (r in 0 until shape[0]) {
            for (c in 0 until shape[1]) {
                assertEquals(view.get(r, c), transposed.get(c, r), "${p.name}: element ($r,$c)")
            }
        }

        // 2. and it describes the same matrix as the physical block-grid permutation the kernels
        // still need. `relayoutPackedWeightForKernels` reorders the blocks to *input-block-major* —
        // block (bI, o) at index `bI * rows + o` — because that is how the packed kernels read a
        // weight, whatever shape it declares (#968/#971; the contract #973 exists to write down).
        // So the permuted bytes are not the row-major encoding of the transposed matrix, and the
        // two paths are not interchangeable until #973 lands: decoded *as block-major*, they carry
        // exactly the values the zero-copy view exposes.
        val physical = ctx.ops.relayoutPackedWeightForKernels(tensor)
        assertEquals(Shape(shape[1], shape[0]), physical.shape, "${p.name}: relayout shape")
        val permutedBytes = (physical.data as PackedBlockStorage).packedData
        assertTrue(
            permutedBytes.contentEquals(GoldenSupport.blockMajor(GoldenSupport.weightBlocks(p, ROWS, BLOCKS_PER_ROW, SEED))),
            "${p.name}: the relayout must produce input-block-major bytes",
        )
        val permuted = build(p, shape, permutedBytes)
        val block = FloatArray(p.blockSize)
        val fromView = FloatArray(shape.volume)
        var i = 0
        for (c in 0 until shape[1]) {
            for (r in 0 until shape[0]) {
                permuted.dequantizeBlock((c / p.blockSize) * ROWS + r, block, 0)
                assertEquals(view.get(r, c), block[c % p.blockSize], "${p.name}: block-major element ($r,$c)")
                fromView[i++] = transposed.get(c, r)
            }
        }
        GoldenSupport.check("transpose/${p.name}", GoldenSupport.digest(fromView))
    }

    @Test fun q4_0() = transposeAgrees(Packed.Q4_0)
    @Test fun q5_0() = transposeAgrees(Packed.Q5_0)
    @Test fun q5_1() = transposeAgrees(Packed.Q5_1)
    @Test fun q8_0() = transposeAgrees(Packed.Q8_0)
    @Test fun q4_K() = transposeAgrees(Packed.Q4_K)
    @Test fun q5_K() = transposeAgrees(Packed.Q5_K)
    @Test fun q6_K() = transposeAgrees(Packed.Q6_K)

    /** Narrowing a transposed packed view still addresses whole blocks — on the axis that carries them. */
    @Test
    fun aTransposedPackedViewStillSlicesByBlock() {
        val p = Packed.Q8_0
        val shape = Shape(ROWS, BLOCKS_PER_ROW * p.blockSize)
        val bytes = GoldenSupport.rowMajor(GoldenSupport.weightBlocks(p, ROWS, BLOCKS_PER_ROW, SEED))
        val data = build(p, shape, bytes)
        val view = data.packedView
        val transposed = view.transpose()

        val head = transposed.narrow(0, 0, p.blockSize)          // one block along the (now leading) block axis
        assertEquals(Shape(p.blockSize, ROWS), head.shape)
        for (c in 0 until p.blockSize) for (r in 0 until ROWS) assertEquals(view.get(r, c), head.get(c, r))

        val rows = transposed.narrow(1, 1, 2)                    // the plain axis narrows freely
        assertEquals(Shape(shape[1], 2), rows.shape)
        assertEquals(view.get(1, 0), rows.get(0, 0))
    }
}
