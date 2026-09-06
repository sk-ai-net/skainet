package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** SKEEP-003 M0-A4: plan arithmetic, fit check and suggestions — from shapes and encodings only. */
class MemoryPlanTest {

    /** Llama-3.2-1B-like geometry. */
    private val llama1b = ModelGeometry(
        layers = 16, heads = 32, kvHeads = 8, headDim = 64, valueDim = 64,
        embeddingLength = 2048, feedForwardLength = 8192, vocabSize = 128_256, trainedContextLength = 131_072,
    )

    private fun q4k(name: String, elements: Long): PlanTensor {
        val f = Format(FP32, TensorEncoding.Q4_K)
        return PlanTensor(name, TensorId.parse(name), f, elements, f.physicalBytes(elements)!!)
    }

    private fun llamaWeights(): List<PlanTensor> = buildList {
        add(q4k("model.embed_tokens.weight", 128_256L * 2048))
        for (n in 0 until 16) {
            add(q4k("model.layers[$n].attn.q_proj.weight", 2048L * 2048)); add(q4k("model.layers[$n].attn.k_proj.weight", 512L * 2048))
            add(q4k("model.layers[$n].attn.v_proj.weight", 512L * 2048)); add(q4k("model.layers[$n].attn.o_proj.weight", 2048L * 2048))
            add(q4k("model.layers[$n].mlp.gate_proj.weight", 8192L * 2048)); add(q4k("model.layers[$n].mlp.up_proj.weight", 8192L * 2048))
            add(q4k("model.layers[$n].mlp.down_proj.weight", 2048L * 8192))
        }
    }

    @Test
    fun kvCacheArithmetic() {
        // layers × 2 (K,V) × ctx × kvHeads × headDim × 2 B = 16 × 2 × 2048 × 8 × 64 × 2 = 64 MiB
        assertEquals(16L * 2048 * 8 * 128, MemoryPlans.kvElements(llama1b, 2048))
        assertEquals(64L * MiB, KvCacheMode.BF16.bytes(MemoryPlans.kvElements(llama1b, 2048)))
        val tq = KvCacheMode.TURBOQUANT_4.bytes(MemoryPlans.kvElements(llama1b, 2048))
        assertTrue(tq in (14L * MiB)..(20L * MiB), "TurboQuant 4-bit KV should be ~¼ of bf16, was ${tq / MiB} MB")
    }

    @Test
    fun forwardSlabScalesWithChunkAndContext() {
        val decode = MemoryPlans.forwardBytes(llama1b, 2048, 1)
        val prefill = MemoryPlans.forwardBytes(llama1b, 2048, 256)
        assertTrue(prefill > decode)
        // decode: 1 token × (4·2048 + 3·8192 + 32·2048) × 4 B + 128256 × 4 B
        assertEquals((4L * 2048 + 3 * 8192 + 32 * 2048) * 4 + 128_256L * 4, decode)
        assertEquals(MemoryPlans.forwardBytes(llama1b, 128, 256), MemoryPlans.forwardBytes(llama1b, 128, 1024)) // chunk capped at ctx
    }

    @Test
    fun planTotalsAndResidency() {
        val input = PlanInput("Llama-3.2-1B-Instruct", "llama", llamaWeights(), llama1b, ctx = 2048)
        val plan = MemoryPlans.plan(input, Budget.of(1300L * MiB))
        assertEquals(input.weights.sumOf { it.bytes }, plan.weightsBytes)
        assertTrue(plan.weightsBytes in (600L * MiB)..(900L * MiB), "Q4_K weights ~0.7 GB, was ${plan.weightsBytes / MiB} MB")
        assertEquals(64L * MiB, plan.kvBytes)
        assertEquals(plan.weightsBytes + plan.kvBytes, plan.residentBytes)
        assertEquals(plan.weightsBytes + plan.kvBytes + plan.forwardBytes + plan.headroomBytes, plan.totalBytes)
        assertEquals(true, plan.fits)
        assertTrue(plan.suggestions().isEmpty())
        assertEquals(listOf("weights", "kv cache", "forward", "heap"), plan.lines.map { it.section })
        assertTrue(plan.lines.first { it.section == "weights" }.resident)
        assertFalse(plan.lines.first { it.section == "forward" }.resident)
        // a weight's allocation is resolved, not assumed (#1143): mapped only when its form asks
        // for MAPPED and the platform can map — these weights carry no form, so they fall to the
        // profile's heap/off-heap threshold, model-lifetime, read-only
        val a = input.weights.first().allocation(PlannerProfile.DESKTOP, StorageCapabilities.FULL)
        assertEquals(MemoryDomain.HOST_OFFHEAP, a.domain); assertEquals(ScopeKind.MODEL, a.scope); assertFalse(a.mutable)
        val mapped = input.weights.first().copy(form = WeightForm(residency = WeightResidency.MAPPED))
            .allocation(PlannerProfile.MOBILE_2GB, StorageCapabilities.FULL)
        assertEquals(MemoryDomain.MMAP_FILE, mapped.domain)
    }

    @Test
    fun doesNotFitGivesAtLeastTwoSuggestionsWithSavings() {
        val input = PlanInput("Llama-3.2-1B-Instruct", "llama", llamaWeights(), llama1b, ctx = 2048)
        val plan = MemoryPlans.plan(input, Budget.of(500L * MiB))
        assertEquals(false, plan.fits)
        val s = plan.suggestions()
        assertTrue(s.size >= 2, "need ≥ 2 suggestions, got $s")
        assertTrue(s.any { it.text == "--kv turboquant" && it.savesBytes == plan.kvBytes - plan.kvBytesAlternate })
        assertTrue(s.any { it.text == "--ctx 1024" && it.savesBytes > 0 })
        assertTrue(s.all { it.savesBytes > 0 })
        val r = plan.render()
        assertTrue(r.contains("✘ does not fit"), r)
        assertTrue(r.contains("suggestions:"), r)
    }

    @Test
    fun turboQuantModeAndBudgetFromAvailableMemory() {
        val input = PlanInput("m", "llama", llamaWeights(), llama1b, ctx = 2048, kvMode = KvCacheMode.TURBOQUANT_4)
        val plan = MemoryPlans.plan(input, Budget.available(2L * GiB))
        assertEquals(2L * GiB - Budget.RESERVE_ANDROID_JVM, plan.budget!!.bytes)
        assertTrue(plan.kvBytes < plan.kvBytesAlternate)
        assertEquals(64L * MiB, plan.kvBytesAlternate)
        assertTrue(plan.render().contains("TurboQuant 4-bit @ ctx 2048"), plan.render())
    }

    @Test
    fun withoutGeometryOrBudget() {
        val input = PlanInput("tiny", "test", listOf(q4k("w", 1024)), geometry = null, ctx = 512)
        val plan = MemoryPlans.plan(input)
        assertEquals(0L, plan.kvBytes); assertEquals(0L, plan.forwardBytes)
        assertNull(plan.fits); assertTrue(plan.suggestions().isEmpty())
        assertEquals(input.weights.single().bytes + MemoryPlans.HEAP_HEADROOM_BYTES, plan.totalBytes)
        assertTrue(plan.render().contains("tiny · test · ctx 512"))
    }

    @Test
    fun unmappedWeightsAreListedNotDropped() {
        val f = Format(FP32, TensorEncoding.Q8_0)
        val input = PlanInput("m", "test", listOf(PlanTensor("blk.0.ffn_gate_exps.weight", null, f, 32, 34)), null, 16)
        assertEquals(listOf("blk.0.ffn_gate_exps.weight"), input.unmappedWeights)
        assertTrue(MemoryPlans.plan(input).render().contains("unmapped tensors (kept, not identified): 1"))
    }

    @Test
    fun formatBytes() {
        assertEquals("512 B", MemoryPlans.formatBytes(512)); assertEquals("4 KB", MemoryPlans.formatBytes(4096))
        assertEquals("68 MB", MemoryPlans.formatBytes(68L * MiB + 100)); assertEquals("1.3 GB", MemoryPlans.formatBytes(1300L * MiB))
    }
}
