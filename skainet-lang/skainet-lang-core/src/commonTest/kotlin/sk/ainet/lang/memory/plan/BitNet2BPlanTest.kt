package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * M2-A1's arithmetic (#1042): **BitNet-b1.58-2B fits in 1.3 GB resident**, computed from the
 * model's geometry the way `skainet-plan` computes it from a GGUF header.
 *
 * The claim M2-A1 makes is about *planning* — what a 2 B-parameter ternary model needs before a
 * byte is read — and that is checkable here, exactly, without a checkpoint. What a real
 * BitNet-2B *measures* on a device belongs to the decode sample in SKaiNET-transformers; this
 * pins the number the planner will predict when it gets there.
 *
 * Geometry: 30 layers, 2560 hidden, 6912 FFN, 20 heads / 5 KV heads (GQA), 128 256 vocab — the
 * published shape of BitNet-b1.58-2B-4T. Ternary linear weights are TQ2_0 (2.0625 bits/element on
 * disk); the embedding table and the output head stay bf16, which is what a b1.58 checkpoint ships
 * and, as the numbers below show, is where most of the resident memory actually goes.
 */
class BitNet2BPlanTest {

    private val mb = 1024L * 1024L
    private val layers = 30
    private val hidden = 2560
    private val ffn = 6912
    private val heads = 20
    private val kvHeads = 5
    private val headDim = hidden / heads
    private val vocab = 128_256

    private fun input(ctx: Int, kvMode: KvCacheMode = KvCacheMode.BF16): PlanInput {
        val ternary = Format(FP32, TensorEncoding.TQ2_0)
        val narrow = Format(BF16, TensorEncoding.Dense(2))
        val weights = ArrayList<PlanTensor>()

        fun ternaryTensor(name: String, elements: Long) {
            weights += PlanTensor(name, null, ternary, elements, ternary.physicalBytes(elements)!!)
        }
        for (l in 0 until layers) {
            // attention: q, k, v, o — GQA narrows k and v
            ternaryTensor("blk.$l.attn_q", hidden.toLong() * hidden)
            ternaryTensor("blk.$l.attn_k", hidden.toLong() * kvHeads * headDim)
            ternaryTensor("blk.$l.attn_v", hidden.toLong() * kvHeads * headDim)
            ternaryTensor("blk.$l.attn_output", hidden.toLong() * hidden)
            // feed-forward: gate, up, down
            ternaryTensor("blk.$l.ffn_gate", hidden.toLong() * ffn)
            ternaryTensor("blk.$l.ffn_up", hidden.toLong() * ffn)
            ternaryTensor("blk.$l.ffn_down", ffn.toLong() * hidden)
        }
        // One embedding table: the output head is tied to it, as the released checkpoint has it.
        // At 128 256 × 2560 in bf16 that single table is 657 MB — more than the entire ternary
        // stack, and the reason a "2-bit model" is not a 500 MB model.
        val embeddingElements = vocab.toLong() * hidden
        weights += PlanTensor("token_embd", null, narrow, embeddingElements, narrow.physicalBytes(embeddingElements)!!)

        val geometry = ModelGeometry(
            layers = layers, heads = heads, kvHeads = kvHeads, headDim = headDim,
            embeddingLength = hidden, feedForwardLength = ffn, vocabSize = vocab,
        )
        return PlanInput("BitNet-b1.58-2B", "bitnet", weights, geometry, ctx, kvMode = kvMode)
    }

    @Test
    fun theTernaryWeightsAreWhatTheEncodingPromises() {
        val plan = MemoryPlans.plan(input(ctx = 2048))
        val ternaryElements = layers.toLong() *
            (2L * hidden * hidden + 2L * hidden * kvHeads * headDim + 3L * hidden * ffn)
        val ternaryBytes = ternaryElements / 256 * 66
        val embeddingBytes = vocab.toLong() * hidden * 2
        assertEquals(ternaryBytes + embeddingBytes, plan.weightsBytes)
        assertEquals(2.0625, ternaryBytes * 8.0 / ternaryElements, 1e-9, "TQ2_0 is 2.0625 bits per element")
        assertTrue(
            ternaryBytes in (500 * mb)..(520 * mb),
            "2 B ternary parameters cost ~512 MB, got ${MemoryPlans.formatBytes(ternaryBytes)}",
        )
        assertTrue(
            embeddingBytes > ternaryBytes,
            "the bf16 embedding table (${MemoryPlans.formatBytes(embeddingBytes)}) outweighs the ternary stack " +
                "(${MemoryPlans.formatBytes(ternaryBytes)}) — that is where a 2-bit model's memory actually goes",
        )
    }

    @Test
    fun m2a1_theModelIsResidentUnderOnePointThreeGigabytes() {
        val limit = (1.3 * 1024 * mb).toLong()
        val quantizedKv = MemoryPlans.plan(input(ctx = 2048, kvMode = KvCacheMode.TURBOQUANT_4))
        assertTrue(
            quantizedKv.residentBytes <= limit,
            "M2-A1: resident (weights + KV) must fit 1.3 GB, was ${MemoryPlans.formatBytes(quantizedKv.residentBytes)}\n" +
                quantizedKv.render(),
        )

        // A bf16 cache fits too — but with 42 MB of headroom against the quantized cache's 148 MB,
        // and the margin closes as the context grows. That thin margin is why the mobile profile
        // quantizes the cache itself rather than leaving it to the caller (#1039).
        val bf16Kv = MemoryPlans.plan(input(ctx = 2048))
        assertTrue(bf16Kv.residentBytes <= limit, bf16Kv.render())
        val bf16Margin = limit - bf16Kv.residentBytes
        val quantizedMargin = limit - quantizedKv.residentBytes
        assertTrue(
            quantizedMargin > bf16Margin * 3,
            "bf16 margin ${MemoryPlans.formatBytes(bf16Margin)} vs quantized ${MemoryPlans.formatBytes(quantizedMargin)}",
        )

        // and at a longer context the bf16 cache does push it over, while the quantized one does not
        val longBf16 = MemoryPlans.plan(input(ctx = 8192))
        val longQuantized = MemoryPlans.plan(input(ctx = 8192, kvMode = KvCacheMode.TURBOQUANT_4))
        assertTrue(longBf16.residentBytes > limit, "ctx 8192 with bf16: ${MemoryPlans.formatBytes(longBf16.residentBytes)}")
        assertTrue(longQuantized.residentBytes <= limit, "ctx 8192 quantized: ${longQuantized.render()}")
    }

    @Test
    fun aTwoGigabyteDeviceCannotHoldThisModelAndTheProfileSaysSoBeforeLoading() {
        // 2 GB total, ~1.5 GB free — the reference board's own numbers. After the profile's 700 MB
        // reserve for the OS and the rest of the app, 800 MB remain, and a 1.2 GB model does not
        // fit in 800 MB however it is staged. The value of the planner is saying that *first*.
        val profiled = PlannerProfile.MOBILE_2GB.plan(input(ctx = 2048), availableBytes = 1500 * mb)
        assertEquals(false, profiled.fits, profiled.render())
        val failure = kotlin.test.assertFailsWith<IllegalStateException> { profiled.requireFits() }
        assertTrue(failure.message!!.contains("mobile-2gb"))
        assertTrue(profiled.plan.suggestions().isNotEmpty(), "and it says what would help")
    }

    @Test
    fun aFourGigabyteDeviceHoldsItWithMappedWeights() {
        val profiled = PlannerProfile.MOBILE_2GB.plan(input(ctx = 2048), availableBytes = 3200 * mb)
        profiled.requireFits()
        val device = DeviceMemory(
            totalRamBytes = 4096 * mb, availableRamBytes = 3200 * mb,
            heapMaxBytes = 512 * mb, heapUsedBytes = 40 * mb, lowMemoryThresholdBytes = 256 * mb,
        )
        // mapped weights: the managed heap carries only the cache, the slab and the headroom
        profiled.requireFits(device)
        assertTrue(profiled.plan.residentBytes <= (1.3 * 1024 * mb).toLong(), profiled.render())
    }

    @Test
    fun m2a1_theSameModelWithHeapStagingDoesNotFitThatDevice() {
        // the counterfactual that makes the mapped path worth having (#921/#922)
        val profiled = PlannerProfile.DESKTOP.plan(input(ctx = 2048), availableBytes = 3200 * mb)
        val phone = DeviceMemory(
            totalRamBytes = 4096 * mb, availableRamBytes = 3200 * mb,
            heapMaxBytes = 512 * mb, heapUsedBytes = 40 * mb, lowMemoryThresholdBytes = 180 * mb,
        )
        val fit = profiled.plan.fitOn(phone, weightsMapped = false)
        assertTrue(!fit.fits, "a 1.2 GB model cannot live on a 512 MB heap:\n${fit.render()}")
        assertEquals("managed heap", fit.blockingPool)
        val mapped = profiled.plan.fitOn(phone, weightsMapped = true)
        assertTrue(mapped.heap.fits, "mapped, the heap is no longer the problem — only the RAM budget is")
    }

    @Test
    fun aLongerContextIsTheFirstThingToGive() {
        val short = MemoryPlans.plan(input(ctx = 2048))
        val long = MemoryPlans.plan(input(ctx = 8192))
        assertTrue(long.kvBytes > short.kvBytes)
        assertEquals(short.weightsBytes, long.weightsBytes, "context length does not change the weights")

        // and quantizing the cache is what the mobile profile does about it
        val quantized = MemoryPlans.plan(input(ctx = 8192, kvMode = KvCacheMode.TURBOQUANT_4))
        assertTrue(quantized.kvBytes < long.kvBytes / 2, "TurboQuant-4 must more than halve a bf16 cache")
    }
}
