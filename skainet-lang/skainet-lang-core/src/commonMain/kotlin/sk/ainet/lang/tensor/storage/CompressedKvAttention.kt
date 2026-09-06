package sk.ainet.lang.tensor.storage

/**
 * Bridge between [KvCacheStore] and the SDPA execution path.
 *
 * This abstraction provides the integration point for compressed K/V
 * in the attention runtime. Instead of modifying the core [TensorOps]
 * interface (which maps to backend-specific fused kernels), this
 * component sits between the model layer and SDPA:
 *
 * 1. **Write path**: Compresses K/V on token append via [storeKeyValue]
 * 2. **Read path**: Dequantizes only required tiles via [loadKeysForAttention]
 *    and [loadValuesForAttention]
 * 3. **Extension point**: Backends can override [DequantStrategy] to fuse
 *    decompression with attention math.
 *
 * Usage in a transformer layer:
 * ```kotlin
 * val bridge = CompressedKvAttention(kvCache)
 * bridge.storeKeyValue(layer, keyProjection, valueProjection)
 * val keys = bridge.loadKeysForAttention(layer)
 * val values = bridge.loadValuesForAttention(layer)
 * // pass keys, values to scaledDotProductAttention
 * ```
 */
public class CompressedKvAttention(
    private val cache: KvCacheStore,
    private val dequantStrategy: DequantStrategy = DequantStrategy.FULL_TILE
) {

    /**
     * Store K/V projections for a new token, compressing as configured.
     *
     * @param layer Layer index
     * @param key   Key projection [numHeads, headDim]
     * @param value Value projection [numHeads, headDim]
     */
    public fun storeKeyValue(layer: Int, key: FloatArray, value: FloatArray) {
        cache.appendToken(layer, key, value)
    }

    /**
     * Load cached keys for attention, dequantizing as needed.
     *
     * When the cache uses compressed encoding, this performs
     * tile-level decompression. The returned array is shaped
     * [numHeads, seqLen, headDim].
     *
     * @param layer    Layer index
     * @param startPos Start of the attention window (inclusive)
     * @param endPos   End of the attention window (exclusive)
     */
    public fun loadKeysForAttention(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen
    ): FloatArray {
        return when (dequantStrategy) {
            DequantStrategy.FULL_TILE -> cache.readKeys(layer, startPos, endPos)
            DequantStrategy.RAW_STORAGE -> {
                // For backends that fuse dequant+attention, return raw storage
                // and let the caller handle it. Fall back to float for now.
                cache.readKeys(layer, startPos, endPos)
            }
        }
    }

    /**
     * Load cached values for attention, dequantizing as needed.
     *
     * @param layer    Layer index
     * @param startPos Start of the attention window (inclusive)
     * @param endPos   End of the attention window (exclusive)
     */
    public fun loadValuesForAttention(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen
    ): FloatArray {
        return when (dequantStrategy) {
            DequantStrategy.FULL_TILE -> cache.readValues(layer, startPos, endPos)
            DequantStrategy.RAW_STORAGE -> {
                cache.readValues(layer, startPos, endPos)
            }
        }
    }

    /**
     * The key window `[startPos, endPos)` as the one or two runs the cache physically holds
     * (#1036, M2-F5) — the pair `WindowedAttention.decodeStep` iterates without copying.
     *
     * For a compressed cache each half is an ordinary view of the decoded window, so a quantized
     * KV store needs no special case here; for the dense ring the halves are views over the ring
     * itself and nothing is copied at all.
     */
    public fun keyWindowForAttention(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen,
    ): sk.ainet.lang.memory.WindowedKV = cache.keyWindow(layer, startPos, endPos)

    /** The value window; see [keyWindowForAttention]. */
    public fun valueWindowForAttention(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen,
    ): sk.ainet.lang.memory.WindowedKV = cache.valueWindow(layer, startPos, endPos)

    /**
     * Load raw [TensorStorage] for keys, preserving the cache's native encoding.
     *
     * This is the zero-copy path for backends that can fuse decompression
     * with attention computation (e.g., Metal fused dequant+SDPA).
     */
    public fun loadKeyStorageRaw(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen
    ): TensorStorage = cache.readKeyStorage(layer, startPos, endPos)

    /**
     * Load raw [TensorStorage] for values, preserving native encoding.
     */
    public fun loadValueStorageRaw(
        layer: Int,
        startPos: Int = 0,
        endPos: Int = cache.currentSeqLen
    ): TensorStorage = cache.readValueStorage(layer, startPos, endPos)

    /** Whether the cache uses compressed (non-Dense) encoding for keys. */
    public val isKeyCompressed: Boolean
        get() = cache.keyEncoding !is TensorEncoding.Dense

    /** Whether the cache uses compressed (non-Dense) encoding for values. */
    public val isValueCompressed: Boolean
        get() = cache.valueEncoding !is TensorEncoding.Dense

    /**
     * Strategy for dequantizing compressed K/V during attention.
     */
    public enum class DequantStrategy {
        /** Decompress the full requested tile to FP32 before attention. */
        FULL_TILE,
        /**
         * Return raw compressed storage — the backend is responsible for
         * fused dequant+attention. Falls back to [FULL_TILE] when no
         * backend fusion is available.
         */
        RAW_STORAGE
    }
}
