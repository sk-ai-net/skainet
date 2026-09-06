package sk.ainet.io

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.MmapTensorSource
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import java.io.File
import java.io.RandomAccessFile

/**
 * JVM and Android [MappedFile]: one read-only `FileChannel.map` over the whole file (available
 * since API 1 — no JNI), tensors served as zero-copy views over its pages.
 *
 * Files larger than 2 GB are refused: a single mapped region is addressed with int offsets.
 * Windowed mapping for bigger files is the follow-up SKEEP-003 §7 names.
 */
public class JvmMappedFile private constructor(
    private val raf: RandomAccessFile,
    private val mmap: MmapTensorSource,
    override val sizeBytes: Long,
) : MappedFile {

    override fun <T : DType> denseFloats(byteOffset: Long, shape: Shape): TensorData<T, Float> =
        mmap.floatTensorAt(byteOffset, shape)

    override fun bytes(byteOffset: Long, length: Int): ByteArray {
        require(byteOffset >= 0 && length >= 0) { "byteOffset and length must be non-negative" }
        require(byteOffset + length <= sizeBytes) { "region [$byteOffset, ${byteOffset + length}) exceeds $sizeBytes bytes" }
        val out = ByteArray(length)
        raf.seek(byteOffset)
        raf.readFully(out)
        return out
    }

    /**
     * GGML block-format payloads as [sk.ainet.lang.tensor.data.BufferPackedTensorData] borrowing a slice
     * of the one file mapping (#1189) — the packed counterpart of [denseFloats]: zero heap bytes,
     * blocks left in canonical row-major file order for the buffer-reading kernels.
     */
    override fun packedTensor(
        byteOffset: Long,
        shape: Shape,
        encoding: sk.ainet.lang.tensor.storage.TensorEncoding,
    ): TensorData<*, *>? = when (encoding) {
        sk.ainet.lang.tensor.storage.TensorEncoding.Q4_K,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q6_K,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q5_K,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q8_0,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q4_0,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q5_0,
        sk.ainet.lang.tensor.storage.TensorEncoding.Q5_1,
        -> {
            val length = checkNotNull(encoding.physicalBytes(shape.volume.toLong())) {
                "${encoding.name} has size-determinate blocks"
            }
            sk.ainet.lang.tensor.data.BufferPackedTensorData(
                shape,
                sk.ainet.lang.memory.DirectBufferStorage.borrow(mmap.byteBufferAt(byteOffset, length)),
                encoding,
            )
        }
        // #1203: the caller (StreamingGgufParametersLoader.i2sTrailerScaleIsMappable) has already
        // confirmed [byteOffset, byteOffset + physicalBytes) is a complete, kernel-ready
        // BITNET_B1_58 buffer -- payload immediately followed by its own trailing FP32 scale, no
        // repack needed at all.
        sk.ainet.lang.tensor.storage.TensorEncoding.BITNET_B1_58 -> {
            val length = checkNotNull(encoding.physicalBytes(shape.volume.toLong())) {
                "${encoding.name} has size-determinate blocks"
            }
            sk.ainet.lang.tensor.data.BitNetB158TensorData.fromStorage(
                shape,
                sk.ainet.lang.memory.DirectBufferStorage.borrow(mmap.byteBufferAt(byteOffset, length)),
            )
        }
        else -> null
    }

    /** Releases the channel; views already handed out keep working (the mapping outlives it). */
    override fun close() {
        try { mmap.close() } finally { raf.close() }
    }

    public companion object {
        /** Map [filePath], or `null` if it is not a readable file or is larger than 2 GB. */
        public fun openOrNull(filePath: String): JvmMappedFile? = try {
            val file = File(filePath)
            if (!file.isFile || !file.canRead() || file.length() > Int.MAX_VALUE) {
                null
            } else {
                val raf = RandomAccessFile(file, "r")
                try {
                    JvmMappedFile(raf, MmapTensorSource.fromChannel(raf.channel), file.length())
                } catch (t: Throwable) {
                    raf.close()
                    throw t
                }
            }
        } catch (e: Exception) {
            null
        }
    }
}

/** JVM/Android: map with [JvmMappedFile]. */
public actual fun openMappedFile(filePath: String): MappedFile? = JvmMappedFile.openOrNull(filePath)
