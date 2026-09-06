package sk.ainet.lang.tensor.storage

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.plan.kvBytesFor
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1077: a KV store declares its `Format` (dtype **and** encoding), so the planner reads the byte
 * width instead of guessing it — the drift that made a dense FP32 ring be planned as bf16.
 */
class KvCacheFormatTest {

    @Test
    fun theDenseStoreDeclaresFp32AndTheDefaultDerivesFromTheEncoding() {
        val store = DefaultKvCacheStore(KvCacheConfig(numLayers = 2, numHeads = 2, headDim = 8, maxSeqLen = 8))
        assertEquals(Format(FP32, TensorEncoding.Dense(4)), store.keyFormat)
        assertEquals(Format(FP32, TensorEncoding.Dense(4)), store.valueFormat)
        assertEquals(4.0, store.keyBytesPerElement)
        assertEquals(4.0, store.valueBytesPerElement)
    }

    @Test
    fun aNarrowFloatRingIsDeclarable() {
        val config = KvCacheConfig(
            numLayers = 2, numHeads = 2, headDim = 8, maxSeqLen = 8,
            keyEncoding = TensorEncoding.Dense(2), valueEncoding = TensorEncoding.Dense(2),
            keyDType = BF16, valueDType = BF16,
        )
        val store = DefaultKvCacheStore(config)
        assertEquals(Format(BF16, TensorEncoding.Dense(2)), store.keyFormat)
        assertEquals(2.0, store.keyBytesPerElement)
        // and the planner's width follows the declaration, not a guess
        assertEquals(2L * 1000, kvBytesFor(store.keyFormat, 1000))
        assertEquals(4L * 1000, kvBytesFor(Format(FP32, TensorEncoding.Dense(4)), 1000))
    }

    @Test
    fun aCompressedStoreReportsItsPackedFormat() {
        val store = KvCacheStore.turboQuant(numLayers = 2, numHeads = 2, headDim = 8, maxSeqLen = 8)
        assertEquals(FP32, store.keyFormat.dtype, "TurboQuant KV is logically FP32")
        assertTrue(store.keyFormat.encoding is TensorEncoding.TurboQuantPolar || store.keyFormat.encoding is TensorEncoding.TurboQuantPolarQjl, store.keyFormat.toString())
        assertTrue(store.keyBytesPerElement < 4.0, "a compressed ring must be narrower than FP32, was ${store.keyBytesPerElement}")
        assertTrue(kvBytesFor(store.keyFormat, 1024) < 4L * 1024)
    }

    @Test
    fun theStoresFormatIsWhatThePlannerShouldUse() {
        val dense = DefaultKvCacheStore(KvCacheConfig(numLayers = 4, numHeads = 2, headDim = 16, maxSeqLen = 64))
        val elements = 4L * 2 * 64 * 16 * 2   // layers × heads × seq × dim × (K+V)
        assertEquals(elements * 4, kvBytesFor(dense.keyFormat, elements), "a dense ring is FP32-wide, not bf16")
    }
}
