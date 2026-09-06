package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32

/**
 * Default KV cache implementation using dense FP32 storage.
 *
 * This is the reference/baseline implementation that stores K/V as
 * uncompressed float arrays. Quantized implementations (Q8_0, TurboQuant)
 * will override [appendToken] and [readKeys]/[readValues] with
 * encode-on-write / decode-on-read paths.
 *
 * Internal layout per layer:
 * - keys:   `FloatArray(numHeads * maxSeqLen * headDim)` — [numHeads, maxSeqLen, headDim]
 * - values: `FloatArray(numHeads * maxSeqLen * headDim)` — [numHeads, maxSeqLen, headDim]
 *
 * Append writes to position [currentSeqLen]; read returns a contiguous slice.
 */
/**
 * @param scope when given, the per-layer K/V backing is allocated in that [sk.ainet.lang.memory.ModelScope]
 *   instead of as plain GC-managed arrays (SKEEP-003 §4.5, PRD M1-F2): the allocation is tracked,
 *   traced (`TraceEvent.Allocation` with the cache's `TensorId`) and released deterministically when
 *   the model closes, which is what lets the memory plan be checked against reality (#1074). The
 *   default keeps today's behaviour exactly.
 */
public class DefaultKvCacheStore @kotlin.jvm.JvmOverloads constructor(
    private val config: KvCacheConfig,
    private val scope: sk.ainet.lang.memory.ModelScope? = null,
    /**
     * Treat the buffer as a **ring** (#1036, M2-F5): appending past [maxSeqLen] overwrites the
     * oldest position instead of failing, and the live window is the newest [maxSeqLen] positions.
     * Off by default — a cache that fills up still throws, exactly as before.
     */
    public val slidingWindow: Boolean = false,
) : KvCacheStore {

    override val numLayers: Int get() = config.numLayers
    override val numHeads: Int get() = config.numHeads
    override val headDim: Int get() = config.headDim
    override val maxSeqLen: Int get() = config.maxSeqLen
    override val keyEncoding: TensorEncoding get() = config.keyEncoding
    override val valueEncoding: TensorEncoding get() = config.valueEncoding

    /** The dense ring holds [KvCacheConfig.keyDType] elements (FP32 unless configured otherwise) — #1077. */
    override val keyFormat: sk.ainet.lang.memory.Format get() = sk.ainet.lang.memory.Format(config.keyDType, config.keyEncoding)

    override val valueFormat: sk.ainet.lang.memory.Format get() = sk.ainet.lang.memory.Format(config.valueDType, config.valueEncoding)
    override val placement: Placement get() = config.placement

    private var _currentSeqLen: Int = 0
    override val currentSeqLen: Int get() = _currentSeqLen

    // Per-layer storage: keys[layer] and values[layer]
    // Each is [numHeads, maxSeqLen, headDim] laid out as contiguous float array.
    // With a ModelScope the arrays come from scope-owned storage (tracked, traced, freed with the
    // model); without one they are plain arrays, exactly as before.
    private val elementsPerLayer: Int = numHeads * maxSeqLen * headDim

    private fun allocateLayer(kind: String, layer: Int): FloatArray {
        val s = scope ?: return FloatArray(elementsPerLayer)
        val storage = s.allocateFloats(
            elementsPerLayer,
            sk.ainet.lang.tensor.TensorId(listOf("kv", "layers[$layer]"), kind),
        )
        val array = storage.floats ?: FloatArray(elementsPerLayer)
        return if (storage.arrayOffset == 0 && array.size == elementsPerLayer) array else FloatArray(elementsPerLayer)
    }

    private val keys: Array<FloatArray> = Array(numLayers) { allocateLayer("k", it) }
    private val values: Array<FloatArray> = Array(numLayers) { allocateLayer("v", it) }

    /** Bytes of K/V backing this store preallocated (`layers × 2 × heads × maxSeqLen × headDim × 4`). */
    public val preallocatedBytes: Long get() = numLayers.toLong() * 2 * elementsPerLayer * 4

    override fun appendToken(layer: Int, key: FloatArray, value: FloatArray) {
        requireLayerIndex(layer)
        check(slidingWindow || _currentSeqLen < maxSeqLen) {
            "KV cache is full: currentSeqLen=$_currentSeqLen, maxSeqLen=$maxSeqLen"
        }
        require(key.size == numHeads * headDim) {
            "Key size mismatch: expected ${numHeads * headDim}, got ${key.size}"
        }
        require(value.size == numHeads * headDim) {
            "Value size mismatch: expected ${numHeads * headDim}, got ${value.size}"
        }

        val pos = slotOf(_currentSeqLen)
        val layerKeys = keys[layer]
        val layerValues = values[layer]

        // Copy each head's slice into the [head, pos, :] position
        for (h in 0 until numHeads) {
            val srcOffset = h * headDim
            val dstOffset = h * maxSeqLen * headDim + pos * headDim
            key.copyInto(layerKeys, dstOffset, srcOffset, srcOffset + headDim)
            value.copyInto(layerValues, dstOffset, srcOffset, srcOffset + headDim)
        }

        // Only increment seqLen when the last layer is written
        if (layer == numLayers - 1) {
            _currentSeqLen++
        }
    }

    override fun readKeys(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(keys[layer], layer, startPos, endPos)
    }

    override fun readValues(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(values[layer], layer, startPos, endPos)
    }

    override fun readKeyStorage(layer: Int, startPos: Int, endPos: Int): TensorStorage {
        return toTensorStorage(readKeys(layer, startPos, endPos), endPos - startPos, keyEncoding)
    }

    override fun readValueStorage(layer: Int, startPos: Int, endPos: Int): TensorStorage {
        return toTensorStorage(readValues(layer, startPos, endPos), endPos - startPos, valueEncoding)
    }

    override fun evict(fromPos: Int) {
        require(fromPos in 0..currentSeqLen) {
            "evict fromPos=$fromPos out of range [0, $currentSeqLen]"
        }
        _currentSeqLen = fromPos
        // Zero out evicted region for safety (prevents stale reads)
        for (layer in 0 until numLayers) {
            for (h in 0 until numHeads) {
                val offset = h * maxSeqLen * headDim + fromPos * headDim
                val count = (maxSeqLen - fromPos) * headDim
                keys[layer].fill(0f, offset, offset + count)
                values[layer].fill(0f, offset, offset + count)
            }
        }
    }

    override fun clear() {
        _currentSeqLen = 0
        for (layer in 0 until numLayers) {
            keys[layer].fill(0f)
            values[layer].fill(0f)
        }
    }

    override fun memoryReport(): KvCacheMemoryReport {
        val elementsPerLayer = numHeads.toLong() * maxSeqLen * headDim
        val logicalBytesPerLayer = elementsPerLayer * 4 // FP32
        return KvCacheMemoryReport(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            currentSeqLen = _currentSeqLen,
            keyEncoding = keyEncoding,
            valueEncoding = valueEncoding,
            placement = placement,
            keyPhysicalBytes = numLayers * logicalBytesPerLayer,
            valuePhysicalBytes = numLayers * logicalBytesPerLayer,
            keyLogicalBytes = numLayers * logicalBytesPerLayer,
            valueLogicalBytes = numLayers * logicalBytesPerLayer
        )
    }

    // --- Internal helpers ---

    // --- the ring (#1036) ---------------------------------------------------------------------

    /** Physical slot of an absolute position. Identity unless this store is a ring. */
    private fun slotOf(position: Int): Int = if (slidingWindow) position % maxSeqLen else position

    /** The oldest absolute position still held — everything before it has been overwritten. */
    public val windowStart: Int
        get() = if (slidingWindow) (_currentSeqLen - maxSeqLen).coerceAtLeast(0) else 0

    /** Storage handles over the per-layer arrays, made once so a window costs no allocation. */
    private val keyStorages by lazy { List(numLayers) { sk.ainet.lang.memory.Storage.Heap.wrap(keys[it]) } }

    private val valueStorages by lazy { List(numLayers) { sk.ainet.lang.memory.Storage.Heap.wrap(values[it]) } }

    override fun keyWindow(layer: Int, from: Int, to: Int): sk.ainet.lang.memory.WindowedKV =
        window(keyStorages[layer], layer, from, to, config.keyDType)

    override fun valueWindow(layer: Int, from: Int, to: Int): sk.ainet.lang.memory.WindowedKV =
        window(valueStorages[layer], layer, from, to, config.valueDType)

    /**
     * `[from, to)` as one or two strided views over the layer's array — zero copies.
     *
     * The layer is laid out `[head, position, dim]`, so a run of positions is a view with the head
     * stride left at `maxSeqLen * headDim`: contiguous per head, strided across heads. When the run
     * crosses the end of the ring it becomes two such views, oldest first.
     */
    private fun window(
        storage: sk.ainet.lang.memory.Storage,
        layer: Int,
        from: Int,
        to: Int,
        dtype: sk.ainet.lang.types.DType,
    ): sk.ainet.lang.memory.WindowedKV {
        requireLayerIndex(layer)
        require(from in windowStart..to) { "window [$from, $to) starts before the ring's oldest position ($windowStart)" }
        require(to <= _currentSeqLen) { "window end $to exceeds currentSeqLen=$_currentSeqLen" }
        val length = to - from
        val startSlot = slotOf(from)
        val firstRun = if (slidingWindow) minOf(length, maxSeqLen - startSlot) else length
        val head = view(storage, startSlot, firstRun, dtype, layer, from)
        val rest = length - firstRun
        return if (rest <= 0) {
            sk.ainet.lang.memory.WindowedKV(head)
        } else {
            sk.ainet.lang.memory.WindowedKV(head, view(storage, 0, rest, dtype, layer, from + firstRun))
        }
    }

    private fun view(
        storage: sk.ainet.lang.memory.Storage,
        startSlot: Int,
        positions: Int,
        dtype: sk.ainet.lang.types.DType,
        layer: Int,
        firstPosition: Int,
    ): sk.ainet.lang.memory.TensorView {
        val shape = Shape(numHeads, positions, headDim)
        val layout = sk.ainet.lang.memory.Layout(
            shape = shape,
            strides = intArrayOf(maxSeqLen * headDim, headDim, 1),
            offsetElements = startSlot.toLong() * headDim,
            elementBytes = dtype.sizeInBytes,
        )
        val id = sk.ainet.lang.tensor.TensorId(listOf("kv", "layers[$layer]"), "window", "from=$firstPosition")
        return sk.ainet.lang.memory.TensorView(shape, sk.ainet.lang.memory.Format(dtype, config.keyEncoding), layout, storage, id)
    }

    private fun readRange(
        layerData: FloatArray,
        layer: Int,
        startPos: Int,
        endPos: Int
    ): FloatArray {
        requireLayerIndex(layer)
        require(startPos in windowStart..endPos) { "Invalid range: startPos=$startPos, endPos=$endPos, oldest held position=$windowStart" }
        require(endPos <= _currentSeqLen) {
            "endPos=$endPos exceeds currentSeqLen=$_currentSeqLen"
        }

        val seqLen = endPos - startPos
        val result = FloatArray(numHeads * seqLen * headDim)
        // A ring's window can cross the end of the buffer: copy it in the one or two runs it
        // occupies, oldest first, so a wrapped read returns positions in order (#1036).
        var written = 0
        var position = startPos
        while (written < seqLen) {
            val slot = slotOf(position)
            val run = if (slidingWindow) minOf(seqLen - written, maxSeqLen - slot) else seqLen - written
            for (h in 0 until numHeads) {
                val srcBase = h * maxSeqLen * headDim + slot * headDim
                val dstBase = h * seqLen * headDim + written * headDim
                layerData.copyInto(result, dstBase, srcBase, srcBase + run * headDim)
            }
            written += run
            position += run
        }
        return result
    }

    private fun toTensorStorage(
        data: FloatArray,
        seqLen: Int,
        encoding: TensorEncoding
    ): TensorStorage {
        // Convert FloatArray to ByteArray for TensorStorage
        val bytes = ByteArray(data.size * 4)
        for (i in data.indices) {
            val bits = data[i].toRawBits()
            bytes[i * 4] = (bits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((bits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((bits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((bits shr 24) and 0xFF).toByte()
        }
        return TensorStorage(
            shape = Shape(numHeads, seqLen, headDim),
            dtype = FP32,
            encoding = encoding,
            buffer = BufferHandle.Owned(bytes),
            placement = placement
        )
    }

    private fun requireLayerIndex(layer: Int) {
        require(layer in 0 until numLayers) {
            "Layer index $layer out of range [0, $numLayers)"
        }
    }
}
