package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * The packed block-layout contract as **fixtures another repository can run** (#973 proposal item
 * 6; #1097).
 *
 * A byte-layout change once shipped as a green-CI hotfix because each repository's tests proved
 * only its own convention: the engine built canonical fixtures, the downstream converter built
 * kernel-native ones, and neither suite crossed the boundary. These builders are published *with
 * the engine*, so a downstream test can assert against the same bytes the engine's own tests
 * assert against — and a change to what "canonical" means fails that test instead of silently
 * shipping.
 *
 * Deliberately in `main`, not a test source set: the whole point is that the artifact a downstream
 * repository already depends on carries them.
 *
 * Every fixture is **three blocks wide** on purpose. At one block per row the two orders coincide,
 * which is exactly the shape that hid #968.
 */
public object PackedLayoutFixtures {

    /** The formats the contract covers — everything with a block geometry and a matmul kernel. */
    public val encodings: List<TensorEncoding> = listOf(
        TensorEncoding.Q4_0, TensorEncoding.Q5_0, TensorEncoding.Q5_1, TensorEncoding.Q8_0,
        TensorEncoding.Q4_K, TensorEncoding.Q5_K, TensorEncoding.Q6_K,
        TensorEncoding.TQ1_0, TensorEncoding.TQ2_0,
    )

    /** Default fixture geometry: four output rows × three blocks per row. */
    public const val ROWS: Int = 4
    public const val BLOCKS_PER_ROW: Int = 3

    /**
     * Canonical (`ROW_MAJOR`) bytes for [encoding]: block `(o, b)` at flat index
     * `o * blocksPerRow + b`, every byte of it set to a value that identifies the block.
     *
     * The content is deliberately trivial and identifying rather than realistic: this fixture
     * exists to pin *where blocks are*, not what they decode to. Decode fidelity is the golden
     * gate's job.
     */
    public fun canonical(
        encoding: TensorEncoding,
        rows: Int = ROWS,
        blocksPerRow: Int = BLOCKS_PER_ROW,
    ): ByteArray {
        val bytesPerBlock = bytesPerBlockOf(encoding)
        val out = ByteArray(rows * blocksPerRow * bytesPerBlock)
        for (o in 0 until rows) {
            for (b in 0 until blocksPerRow) {
                val base = (o * blocksPerRow + b) * bytesPerBlock
                for (i in 0 until bytesPerBlock) out[base + i] = blockTag(o, b)
            }
        }
        return out
    }

    /** The same weight in the order the kernels read (`INPUT_BLOCK_MAJOR`): block `(o, b)` at `b * rows + o`. */
    public fun kernelOrder(
        encoding: TensorEncoding,
        rows: Int = ROWS,
        blocksPerRow: Int = BLOCKS_PER_ROW,
    ): ByteArray = PackedWeights.toKernelOrder(canonical(encoding, rows, blocksPerRow), rows, blocksPerRow, bytesPerBlockOf(encoding))

    /** The byte every byte of block `(o, b)` carries — an identity a test can assert on. */
    public fun blockTag(row: Int, block: Int): Byte = (row * 16 + block).toByte()

    /** Bytes per block of [encoding], from its own descriptor. */
    public fun bytesPerBlockOf(encoding: TensorEncoding): Int {
        val spec = encoding.blockSpec ?: throw IllegalArgumentException("$encoding is not block-structured")
        require(!spec.isPerTensor) { "${encoding.name} has no fixed block size; it is a per-tensor encoding" }
        return spec.bytesPerBlock
    }

    /**
     * Check that [bytes] holds the fixture's blocks in the given order — the assertion a downstream
     * test runs against its own converter's output.
     *
     * @return `null` when it agrees, or a description of the first block that is in the wrong place
     */
    public fun disagreement(
        bytes: ByteArray,
        encoding: TensorEncoding,
        kernelOrder: Boolean,
        rows: Int = ROWS,
        blocksPerRow: Int = BLOCKS_PER_ROW,
    ): String? {
        val bytesPerBlock = bytesPerBlockOf(encoding)
        val required = rows * blocksPerRow * bytesPerBlock
        if (bytes.size < required) return "expected at least $required bytes, got ${bytes.size}"
        for (o in 0 until rows) {
            for (b in 0 until blocksPerRow) {
                val index = if (kernelOrder) b * rows + o else o * blocksPerRow + b
                val actual = bytes[index * bytesPerBlock]
                val expected = blockTag(o, b)
                if (actual != expected) {
                    return "block ($o, $b) should be at flat index $index " +
                        "(${if (kernelOrder) "input-block-major" else "canonical"}), found tag $actual instead of $expected"
                }
            }
        }
        return null
    }
}
