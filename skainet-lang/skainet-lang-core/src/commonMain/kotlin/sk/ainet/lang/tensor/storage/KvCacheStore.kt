package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantConfig
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantPreset
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantPresets

/**
 * Dedicated KV-cache storage abstraction for inference.
 *
 * Unlike generic [TensorStorage], a KV cache is **append-friendly** and
 * **role-aware**: keys and values may use different encodings and bit budgets.
 * The cache is addressed by (layer, head, position) and supports compressed
 * block storage for quantized formats (Q4_K, Q8_0, TurboQuant, etc.).
 *
 * Backends and attention kernels interact with the cache through this
 * interface rather than managing raw tensors directly. This allows:
 * - Compressed K/V writes on token append
 * - Tile-level dequantization on read (only the needed range)
 * - Asymmetric K/V policies (e.g., Q8_0 for keys, 4-bit for values)
 * - Backend-specific fused dequant+attention paths
 */
private fun bytesPerElement(format: sk.ainet.lang.memory.Format): Double {
    val probe = 1024L
    val bytes = format.physicalBytes(probe) ?: return format.dtype.sizeInBytes.toDouble()
    return bytes.toDouble() / probe
}

/** Average bytes per element of [format]; fractional for packed encodings. */
internal fun kvBytesPerElement(format: sk.ainet.lang.memory.Format): Double {
    val probe = 1024L
    val bytes = format.physicalBytes(probe) ?: return format.dtype.sizeInBytes.toDouble()
    return bytes.toDouble() / probe
}

public interface KvCacheStore {

    /** Number of transformer layers in this cache. */
    public val numLayers: Int

    /** Number of KV heads per layer. */
    public val numHeads: Int

    /** Dimension per head. */
    public val headDim: Int

    /** Maximum sequence length this cache can hold. */
    public val maxSeqLen: Int

    /** Current number of tokens stored in the cache. */
    public val currentSeqLen: Int

    /** Encoding used for key storage. */
    public val keyEncoding: TensorEncoding

    /** Encoding used for value storage. */
    public val valueEncoding: TensorEncoding

    /**
     * What the keys *are* and how they are stored: `Format(FP32, Dense(4))` for a dense ring,
     * `Format(BF16, Dense(2))` for a narrow-float one, `Format(FP32, TurboQuantPolar(4, 128))` for a
     * compressed one (#1077, SKEEP-003 §0 *Format*).
     *
     * The default derives from [keyEncoding] and assumes FP32 — what every store held before this
     * existed — so no implementation breaks; a store that keeps something else overrides it. The
     * memory planner reads this instead of guessing a byte width, and guessing is what made a dense
     * FP32 ring be planned as bf16 and understated by 2× (#1074 caught it; this makes it impossible).
     */
    public val keyFormat: sk.ainet.lang.memory.Format
        get() = sk.ainet.lang.memory.Format(sk.ainet.lang.types.FP32, keyEncoding)

    /** What the values are and how they are stored; see [keyFormat]. */
    public val valueFormat: sk.ainet.lang.memory.Format
        get() = sk.ainet.lang.memory.Format(sk.ainet.lang.types.FP32, valueEncoding)

    /** Bytes one key element occupies under [keyFormat] (fractional for sub-byte encodings). */
    public val keyBytesPerElement: Double get() = kvBytesPerElement(keyFormat)

    /** Bytes one value element occupies under [valueFormat]. */
    public val valueBytesPerElement: Double get() = kvBytesPerElement(valueFormat)

    /**
     * The attention window `[from, to)` of this layer's keys, as the one or two runs it physically
     * occupies (#1036, M2-F5). A ring that has wrapped holds its newest positions in two runs; the
     * pair is handed to attention instead of being copied together first.
     *
     * The default implementation copies the range once through [readKeys], which is correct for
     * every store; [DefaultKvCacheStore] overrides it with zero-copy views over the ring itself.
     */
    public fun keyWindow(layer: Int, from: Int = 0, to: Int = currentSeqLen): sk.ainet.lang.memory.WindowedKV =
        copiedWindow(readKeys(layer, from, to), to - from, keyFormat.dtype)

    /** The attention window of this layer's values; see [keyWindow]. */
    public fun valueWindow(layer: Int, from: Int = 0, to: Int = currentSeqLen): sk.ainet.lang.memory.WindowedKV =
        copiedWindow(readValues(layer, from, to), to - from, valueFormat.dtype)

    /** A single-run window over an already-materialized `[heads, positions, headDim]` array. */
    private fun copiedWindow(
        values: FloatArray,
        positions: Int,
        dtype: sk.ainet.lang.types.DType,
    ): sk.ainet.lang.memory.WindowedKV = sk.ainet.lang.memory.WindowedKV(
        sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.Storage.Heap.wrap(values),
            sk.ainet.lang.tensor.Shape(numHeads, positions, headDim),
            dtype,
        ),
    )

    /** Placement intent for the cache buffers. */
    public val placement: Placement

    /**
     * Append a single token's K/V projections for one layer.
     *
     * The runtime calls this once per layer per generated token. The cache
     * is responsible for encoding/compressing the data according to
     * [keyEncoding] and [valueEncoding].
     *
     * @param layer  Layer index (0-based)
     * @param key    Key projection [numHeads, headDim] as float
     * @param value  Value projection [numHeads, headDim] as float
     * @throws IllegalStateException if the cache is full ([currentSeqLen] >= [maxSeqLen])
     */
    public fun appendToken(layer: Int, key: FloatArray, value: FloatArray)

    /**
     * Read cached keys for a layer, dequantized to float.
     *
     * Returns the key cache for positions `[startPos, endPos)` as a
     * contiguous float array shaped [numHeads, (endPos - startPos), headDim].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive), defaults to [currentSeqLen]
     */
    public fun readKeys(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): FloatArray

    /**
     * Read cached values for a layer, dequantized to float.
     *
     * Returns the value cache for positions `[startPos, endPos)` as a
     * contiguous float array shaped [numHeads, (endPos - startPos), headDim].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive), defaults to [currentSeqLen]
     */
    public fun readValues(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): FloatArray

    /**
     * Read raw (possibly compressed) key storage for a layer as [TensorStorage].
     *
     * This is the zero-copy path for backends that can fuse dequantization
     * with attention computation. Returns storage with the cache's native
     * [keyEncoding].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive)
     */
    public fun readKeyStorage(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): TensorStorage

    /**
     * Read raw (possibly compressed) value storage for a layer as [TensorStorage].
     *
     * @param layer    Layer index
     * @param startPos First token position (inclusive)
     * @param endPos   Last token position (exclusive)
     */
    public fun readValueStorage(layer: Int, startPos: Int = 0, endPos: Int = currentSeqLen): TensorStorage

    /**
     * Evict all cached tokens from position [fromPos] onward.
     *
     * Used for sequence truncation or speculative decoding rollback.
     * After eviction, [currentSeqLen] becomes [fromPos].
     */
    public fun evict(fromPos: Int)

    /** Reset the cache, clearing all stored tokens. */
    public fun clear()

    /**
     * Memory report for the entire cache.
     */
    public fun memoryReport(): KvCacheMemoryReport

    public companion object {
        /**
         * Create an uncompressed FP32 KV cache (baseline).
         *
         * Use this when you don't need compression or as a reference
         * for quality comparison.
         */
        public fun dense(
            numLayers: Int,
            numHeads: Int,
            headDim: Int,
            maxSeqLen: Int
        ): KvCacheStore = DefaultKvCacheStore(
            KvCacheConfig.dense(numLayers, numHeads, headDim, maxSeqLen)
        )

        /**
         * Create a TurboQuant-compressed KV cache from a named preset.
         *
         * Available presets: "safe-lowbit", "balanced", "experimental-max".
         *
         * Example:
         * ```kotlin
         * val cache = KvCacheStore.turboQuant("balanced", numLayers=32, numHeads=32, headDim=128, maxSeqLen=4096)
         * ```
         *
         * @param preset    Preset name (see [TurboQuantPresets.availablePresets])
         * @param numLayers Number of transformer layers
         * @param numHeads  Number of KV heads per layer
         * @param headDim   Dimension per head
         * @param maxSeqLen Maximum sequence length
         */
        public fun turboQuant(
            preset: String,
            numLayers: Int,
            numHeads: Int,
            headDim: Int,
            maxSeqLen: Int
        ): KvCacheStore {
            val resolved = TurboQuantPresets.forModel(preset, numLayers, numHeads, headDim, maxSeqLen)
            return fromPreset(resolved)
        }

        /**
         * Create a TurboQuant-compressed KV cache with custom bit budgets.
         *
         * Example:
         * ```kotlin
         * // 8-bit keys, 4-bit values (safe-lowbit style)
         * val cache = KvCacheStore.turboQuant(
         *     numLayers=32, numHeads=32, headDim=128, maxSeqLen=4096,
         *     keyBits=8, valueBits=4
         * )
         * ```
         */
        public fun turboQuant(
            numLayers: Int,
            numHeads: Int,
            headDim: Int,
            maxSeqLen: Int,
            keyBits: Int = 4,
            valueBits: Int = 4,
            useQjl: Boolean = false
        ): KvCacheStore {
            val config = KvCacheConfig(
                numLayers = numLayers,
                numHeads = numHeads,
                headDim = headDim,
                maxSeqLen = maxSeqLen,
                keyEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = keyBits),
                valueEncoding = TensorEncoding.TurboQuantPolar(bitsPerElement = valueBits)
            )
            val keyConfig = if (useQjl) TurboQuantConfig.polarPlusQjl(bits = keyBits)
                            else TurboQuantConfig.polarOnly(bits = keyBits)
            val valueConfig = if (useQjl) TurboQuantConfig.polarPlusQjl(bits = valueBits)
                              else TurboQuantConfig.polarOnly(bits = valueBits)
            return TurboQuantKvCacheStore(config, keyConfig, valueConfig)
        }

        /**
         * Create a KV cache from a [TurboQuantPreset].
         */
        public fun fromPreset(preset: TurboQuantPreset): KvCacheStore {
            val keyConfig = preset.keyQuantConfig ?: TurboQuantConfig.polarOnly(bits = 4)
            val valueConfig = preset.valueQuantConfig ?: TurboQuantConfig.polarOnly(bits = 4)
            return TurboQuantKvCacheStore(preset.cacheConfig, keyConfig, valueConfig)
        }
    }
}

/**
 * Configuration for asymmetric K/V encoding policies.
 *
 * Keys are often more quality-sensitive than values, so different
 * bit budgets may be appropriate. For example:
 * - safe-lowbit: Q8_0 keys + 4-bit values
 * - balanced:    4-bit keys + 4-bit values
 */
public data class KvCacheConfig(
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val maxSeqLen: Int,
    val keyEncoding: TensorEncoding = TensorEncoding.Dense(4),
    val valueEncoding: TensorEncoding = TensorEncoding.Dense(4),
    val placement: Placement = Placement.CPU_HEAP,
    /**
     * The dtype the key ring stores; with [keyEncoding] it forms the store's `keyFormat` (#1077).
     * `FP32` is what the dense store has always held; `BF16`/`FP16` halve the ring.
     */
    val keyDType: sk.ainet.lang.types.DType = sk.ainet.lang.types.FP32,
    /** The dtype the value ring stores; see [keyDType]. */
    val valueDType: sk.ainet.lang.types.DType = sk.ainet.lang.types.FP32,
) {
    init {
        require(numLayers > 0) { "numLayers must be positive: $numLayers" }
        require(numHeads > 0) { "numHeads must be positive: $numHeads" }
        require(headDim > 0) { "headDim must be positive: $headDim" }
        require(maxSeqLen > 0) { "maxSeqLen must be positive: $maxSeqLen" }
    }

    public companion object {
        /** Uncompressed FP32 cache (baseline). */
        public fun dense(numLayers: Int, numHeads: Int, headDim: Int, maxSeqLen: Int): KvCacheConfig =
            KvCacheConfig(numLayers, numHeads, headDim, maxSeqLen)

        /** Q8_0-compressed cache for both K and V. */
        public fun q8(numLayers: Int, numHeads: Int, headDim: Int, maxSeqLen: Int): KvCacheConfig =
            KvCacheConfig(
                numLayers, numHeads, headDim, maxSeqLen,
                keyEncoding = TensorEncoding.Q8_0,
                valueEncoding = TensorEncoding.Q8_0
            )
    }
}

/**
 * Memory report for a KV cache instance.
 */
public data class KvCacheMemoryReport(
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val maxSeqLen: Int,
    val currentSeqLen: Int,
    val keyEncoding: TensorEncoding,
    val valueEncoding: TensorEncoding,
    val placement: Placement,
    val keyPhysicalBytes: Long,
    val valuePhysicalBytes: Long,
    val keyLogicalBytes: Long,
    val valueLogicalBytes: Long
) {
    val totalPhysicalBytes: Long get() = keyPhysicalBytes + valuePhysicalBytes
    val totalLogicalBytes: Long get() = keyLogicalBytes + valueLogicalBytes
    val compressionRatio: Double
        get() = if (totalPhysicalBytes > 0) totalLogicalBytes.toDouble() / totalPhysicalBytes else 1.0
    val utilizationRatio: Double
        get() = if (maxSeqLen > 0) currentSeqLen.toDouble() / maxSeqLen else 0.0
}
