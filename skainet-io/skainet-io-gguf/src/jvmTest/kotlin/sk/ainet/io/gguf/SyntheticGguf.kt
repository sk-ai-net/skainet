package sk.ainet.io.gguf

import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.tensor.storage.TensorEncoding
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * Builder for synthetic multi-tensor GGUF v3 files used by the #782 tests.
 *
 * Payloads are seeded-pseudo-random block data with the per-block scale fields
 * patched to small, finite half-precision values, so dequantization never
 * produces NaN/Inf and bit-exact comparisons are meaningful.
 */
object SyntheticGguf {

    data class TestTensor(
        val name: String,
        val type: GGMLQuantizationType,
        val elementCount: Long,
        val data: ByteArray,
        /**
         * Dimensions in GGUF `ne` order (fastest-varying first). Defaults to rank 1, which keeps
         * element order unambiguous; a 2-D weight is written `[in, out]`, as a real file has it
         * (#1098).
         */
        val dims: List<Long> = listOf(elementCount),
    )

    /** Bytes per block / elements per block for [type], from [GGML_QUANT_SIZES]. */
    private fun blockLayout(type: GGMLQuantizationType): Pair<Int, Int> {
        val (blockElems, blockBytes) = GGML_QUANT_SIZES.getValue(type)
        return blockBytes to blockElems
    }

    /**
     * Build a tensor of [elements] logical elements (must be a multiple of the
     * format's block size) with deterministic pseudo-random payload.
     */
    fun tensor(name: String, type: GGMLQuantizationType, elements: Int, seed: Int = name.hashCode()): TestTensor {
        val rnd = Random(seed)
        val bytes: ByteArray = when (type) {
            GGMLQuantizationType.F32 -> {
                val buf = ByteBuffer.allocate(elements * 4).order(ByteOrder.LITTLE_ENDIAN)
                repeat(elements) { buf.putFloat((rnd.nextFloat() - 0.5f) * 4f) }
                buf.array()
            }
            GGMLQuantizationType.F16 -> {
                val buf = ByteBuffer.allocate(elements * 2).order(ByteOrder.LITTLE_ENDIAN)
                // Normal, finite halves: exponent in 12..17, random sign+mantissa.
                repeat(elements) {
                    val bits = ((12 + rnd.nextInt(6)) shl 10) or rnd.nextInt(0x400) or
                        (if (rnd.nextBoolean()) 0x8000 else 0)
                    buf.putShort(bits.toShort())
                }
                buf.array()
            }
            GGMLQuantizationType.BF16 -> {
                val buf = ByteBuffer.allocate(elements * 2).order(ByteOrder.LITTLE_ENDIAN)
                // Normal, finite bf16: exponent in 120..127.
                repeat(elements) {
                    val bits = ((120 + rnd.nextInt(8)) shl 7) or rnd.nextInt(0x80) or
                        (if (rnd.nextBoolean()) 0x8000 else 0)
                    buf.putShort(bits.toShort())
                }
                buf.array()
            }
            GGMLQuantizationType.TQ1_0, GGMLQuantizationType.TQ2_0 -> ternary(name, type, elements, seed).second
            else -> {
                val (blockBytes, blockElems) = blockLayout(type)
                require(elements % blockElems == 0) {
                    "$name: $elements elements is not a multiple of ${type.name} block size $blockElems"
                }
                val blockCount = elements / blockElems
                val bytes = rnd.nextBytes(blockCount * blockBytes)
                patchScales(bytes, type, blockCount, blockBytes, rnd)
                bytes
            }
        }
        return TestTensor(name, type, elements.toLong(), bytes)
    }

    /**
     * A ternary tensor (`TQ1_0` / `TQ2_0`) and the exact values it encodes (#1033).
     *
     * Values are drawn from `{-0.5, 0, +0.5}`, so every block's absmax is 0.5 — exactly
     * representable in FP16 — and the file round-trips to the values bit for bit. The bytes come
     * from [TernaryCodec], the same reference encoder the decoder is defined against, so this
     * fixture exercises the real GGML layout (interleave included) rather than a plausible one.
     */
    fun ternary(
        name: String,
        type: GGMLQuantizationType,
        elements: Int,
        seed: Int = name.hashCode(),
    ): Triple<TestTensor, ByteArray, FloatArray> {
        val encoding = when (type) {
            GGMLQuantizationType.TQ1_0 -> TensorEncoding.TQ1_0
            GGMLQuantizationType.TQ2_0 -> TensorEncoding.TQ2_0
            else -> error("$type is not a ternary GGML type")
        }
        val rnd = Random(seed)
        val values = FloatArray(elements) { (rnd.nextInt(3) - 1) * 0.5f }
        val bytes = TernaryCodec.encode(encoding, values)
        return Triple(TestTensor(name, type, elements.toLong(), bytes), bytes, values)
    }

    /**
     * Overwrite the fp16 scale fields of every block with small finite normals
     * (raw payload bytes elsewhere are valid codes for every format).
     */
    private fun patchScales(
        bytes: ByteArray,
        type: GGMLQuantizationType,
        blockCount: Int,
        blockBytes: Int,
        rnd: Random,
    ) {
        fun putHalf(offset: Int, base: Int) {
            val bits = base + rnd.nextInt(0x100) // small normal half range
            bytes[offset] = (bits and 0xFF).toByte()
            bytes[offset + 1] = ((bits shr 8) and 0xFF).toByte()
        }
        for (b in 0 until blockCount) {
            val base = b * blockBytes
            when (type) {
                // d @0
                GGMLQuantizationType.Q4_0,
                GGMLQuantizationType.Q5_0,
                GGMLQuantizationType.Q8_0 -> putHalf(base, 0x3400)
                // d @0, m/dmin @2
                GGMLQuantizationType.Q5_1,
                GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K -> {
                    putHalf(base, 0x3400)
                    putHalf(base + 2, 0x2C00)
                }
                // Q6_K: d is the *last* two bytes of the 210-byte block
                GGMLQuantizationType.Q6_K -> putHalf(base + 208, 0x3400)
                else -> error("patchScales: unhandled $type")
            }
        }
    }

    /** Write a GGUF v3 file containing [tensors] (32-byte aligned data section). */
    fun write(vararg tensors: TestTensor): File {
        val file = File.createTempFile("synthetic_", ".gguf")
        file.deleteOnExit()

        // ---- header + KV + tensor-info section
        val head = ByteBuffer.allocate(64 * 1024).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747) // "GGUF"
        head.putInt(3)
        head.putLong(tensors.size.toLong())
        head.putLong(1) // KV count

        val key = "general.architecture".encodeToByteArray()
        head.putLong(key.size.toLong())
        head.put(key)
        head.putInt(GGUFValueType.STRING.value)
        val value = "test".encodeToByteArray()
        head.putLong(value.size.toLong())
        head.put(value)

        var dataOffset = 0L
        for (t in tensors) {
            val name = t.name.encodeToByteArray()
            head.putLong(name.size.toLong())
            head.put(name)
            head.putInt(t.dims.size)
            for (d in t.dims) head.putLong(d)
            head.putInt(t.type.value)
            head.putLong(dataOffset)
            // every payload here is already a multiple of 32 bytes or padded below
            dataOffset += padded(t.data.size).toLong()
        }
        val padding = (32 - (head.position() % 32)) % 32
        repeat(padding) { head.put(0) }

        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            for (t in tensors) {
                raf.write(t.data)
                repeat(padded(t.data.size) - t.data.size) { raf.write(0) }
            }
        }
        return file
    }

    private fun padded(size: Int): Int = ((size + 31) / 32) * 32
}
