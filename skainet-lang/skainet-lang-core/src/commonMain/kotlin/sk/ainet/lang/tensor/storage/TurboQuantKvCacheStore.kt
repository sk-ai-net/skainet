package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantBlock
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantCodec
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantConfig
import sk.ainet.lang.tensor.ops.turboquant.RandomRotation

/**
 * KV cache store with TurboQuant compression.
 *
 * Compresses K/V projections on write using TurboQuant and decompresses
 * on read. Supports asymmetric K/V policies (different bit budgets and
 * variants for keys vs values).
 *
 * Each token's K/V projection per head is stored as a [TurboQuantBlock].
 * This gives fine-grained control: different layers/heads could
 * potentially use different configurations (though this implementation
 * uses uniform config).
 */
public class TurboQuantKvCacheStore(
    private val config: KvCacheConfig,
    private val keyConfig: TurboQuantConfig,
    private val valueConfig: TurboQuantConfig
) : KvCacheStore {

    override val numLayers: Int get() = config.numLayers
    override val numHeads: Int get() = config.numHeads
    override val headDim: Int get() = config.headDim
    override val maxSeqLen: Int get() = config.maxSeqLen
    override val keyEncoding: TensorEncoding get() = config.keyEncoding
    override val valueEncoding: TensorEncoding get() = config.valueEncoding

    /** Logically FP32 values held as TurboQuant codes — the encoding the config carries (#1077). */
    override val keyFormat: sk.ainet.lang.memory.Format get() = sk.ainet.lang.memory.Format(sk.ainet.lang.types.FP32, keyEncoding)

    override val valueFormat: sk.ainet.lang.memory.Format get() = sk.ainet.lang.memory.Format(sk.ainet.lang.types.FP32, valueEncoding)
    override val placement: Placement get() = config.placement

    private var _currentSeqLen: Int = 0
    override val currentSeqLen: Int get() = _currentSeqLen

    // Compressed storage: [layer][position] -> array of TurboQuantBlock (one per head)
    private val keyBlocks: Array<Array<Array<TurboQuantBlock?>>> = Array(numLayers) {
        Array(maxSeqLen) { arrayOfNulls(numHeads) }
    }
    private val valueBlocks: Array<Array<Array<TurboQuantBlock?>>> = Array(numLayers) {
        Array(maxSeqLen) { arrayOfNulls(numHeads) }
    }

    override fun appendToken(layer: Int, key: FloatArray, value: FloatArray) {
        requireLayerIndex(layer)
        check(_currentSeqLen < maxSeqLen) {
            "KV cache is full: currentSeqLen=$_currentSeqLen, maxSeqLen=$maxSeqLen"
        }
        require(key.size == numHeads * headDim) {
            "Key size mismatch: expected ${numHeads * headDim}, got ${key.size}"
        }
        require(value.size == numHeads * headDim) {
            "Value size mismatch: expected ${numHeads * headDim}, got ${value.size}"
        }

        val pos = _currentSeqLen

        for (h in 0 until numHeads) {
            val headKey = key.copyOfRange(h * headDim, (h + 1) * headDim)
            val headValue = value.copyOfRange(h * headDim, (h + 1) * headDim)

            val keySeed = RandomRotation.seedFor(layer, h, pos)
            val valueSeed = keySeed xor 0x5A5A5A5A.toInt()

            keyBlocks[layer][pos][h] = TurboQuantCodec.encode(
                headKey, keyConfig.copy(seed = keySeed)
            )
            valueBlocks[layer][pos][h] = TurboQuantCodec.encode(
                headValue, valueConfig.copy(seed = valueSeed)
            )
        }

        if (layer == numLayers - 1) {
            _currentSeqLen++
        }
    }

    override fun readKeys(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(keyBlocks, layer, startPos, endPos)
    }

    override fun readValues(layer: Int, startPos: Int, endPos: Int): FloatArray {
        return readRange(valueBlocks, layer, startPos, endPos)
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
        for (layer in 0 until numLayers) {
            for (pos in fromPos until maxSeqLen) {
                for (h in 0 until numHeads) {
                    keyBlocks[layer][pos][h] = null
                    valueBlocks[layer][pos][h] = null
                }
            }
        }
        _currentSeqLen = fromPos
    }

    override fun clear() {
        _currentSeqLen = 0
        for (layer in 0 until numLayers) {
            for (pos in 0 until maxSeqLen) {
                for (h in 0 until numHeads) {
                    keyBlocks[layer][pos][h] = null
                    valueBlocks[layer][pos][h] = null
                }
            }
        }
    }

    override fun memoryReport(): KvCacheMemoryReport {
        var keyBytes = 0L
        var valueBytes = 0L
        for (layer in 0 until numLayers) {
            for (pos in 0 until _currentSeqLen) {
                for (h in 0 until numHeads) {
                    keyBytes += keyBlocks[layer][pos][h]?.sizeInBytes ?: 0
                    valueBytes += valueBlocks[layer][pos][h]?.sizeInBytes ?: 0
                }
            }
        }
        val logicalPerLayer = numHeads.toLong() * _currentSeqLen * headDim * 4
        return KvCacheMemoryReport(
            numLayers = numLayers,
            numHeads = numHeads,
            headDim = headDim,
            maxSeqLen = maxSeqLen,
            currentSeqLen = _currentSeqLen,
            keyEncoding = keyEncoding,
            valueEncoding = valueEncoding,
            placement = placement,
            keyPhysicalBytes = keyBytes,
            valuePhysicalBytes = valueBytes,
            keyLogicalBytes = numLayers * logicalPerLayer,
            valueLogicalBytes = numLayers * logicalPerLayer
        )
    }

    // --- Internal ---

    private fun readRange(
        blocks: Array<Array<Array<TurboQuantBlock?>>>,
        layer: Int,
        startPos: Int,
        endPos: Int
    ): FloatArray {
        requireLayerIndex(layer)
        require(startPos in 0..endPos) { "Invalid range: startPos=$startPos, endPos=$endPos" }
        require(endPos <= _currentSeqLen) { "endPos=$endPos exceeds currentSeqLen=$_currentSeqLen" }

        val seqLen = endPos - startPos
        // Output: [numHeads, seqLen, headDim]
        val result = FloatArray(numHeads * seqLen * headDim)

        for (h in 0 until numHeads) {
            for (p in startPos until endPos) {
                val block = blocks[layer][p][h]
                    ?: error("Missing block at layer=$layer, pos=$p, head=$h")
                val decoded = TurboQuantCodec.decode(block)
                val dstOffset = h * seqLen * headDim + (p - startPos) * headDim
                decoded.copyInto(result, dstOffset)
            }
        }
        return result
    }

    private fun toTensorStorage(data: FloatArray, seqLen: Int, encoding: TensorEncoding): TensorStorage {
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
        require(layer in 0 until numLayers) { "Layer $layer out of range [0, $numLayers)" }
    }
}
