
package sk.ainet.apps.plan

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.ModelGeometry
import sk.ainet.lang.memory.plan.PlanInput
import sk.ainet.lang.memory.plan.PlanTensor
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SkainetPlanTest {

    @Test
    fun parsesSizes() {
        assertEquals(1L shl 30, parseBytes("1G")); assertEquals((1.3 * (1L shl 30)).toLong(), parseBytes("1.3G"))
        assertEquals(900L shl 20, parseBytes("900M")); assertEquals(900L shl 20, parseBytes("900MB"))
        assertEquals(64L shl 10, parseBytes("64k")); assertEquals(123456L, parseBytes("123456"))
        assertFailsWith<IllegalArgumentException> { parseBytes("lots") }
    }

    @Test
    fun globMatching() {
        val re = globToRegex("model.layers[3].*")
        assertTrue(re.matches("model.layers[3].attn.q_proj.weight"))
        assertTrue(!re.matches("model.layers[13].attn.q_proj.weight"))
        assertTrue(globToRegex("*.weight").matches("model.norm.weight"))
        assertTrue(globToRegex("model.layers[?].attn.*").matches("model.layers[7].attn.k_proj.bias"))
    }

    @Test
    fun listRendersIdFormatShapeAndSourceName() {
        val f = Format(FP32, TensorEncoding.Q4_K)
        val w = PlanTensor("blk.3.attn_q.weight", TensorId.parse("model.layers[3].attn.q_proj.weight"), f, 2048L * 2048, f.physicalBytes(2048L * 2048)!!)
        val other = PlanTensor("blk.4.attn_q.weight", TensorId.parse("model.layers[4].attn.q_proj.weight"), f, 2048L * 2048, f.physicalBytes(2048L * 2048)!!)
        val g = ModelGeometry(16, 32, 8, 64, 64, 2048, 8192, 128_256)
        val plan = MemoryPlans.plan(PlanInput("m", "llama", listOf(w, other), g, 2048), Budget.of(1300L shl 20))
        val out = renderList(plan, "model.layers[3].*")
        assertTrue(out.contains("tensors matching 'model.layers[3].*': 1"), out)
        assertTrue(out.contains("model.layers[3].attn.q_proj.weight"), out)
        assertTrue(out.contains("Float32/Q4_K"), out)
        assertTrue(out.contains("← blk.3.attn_q.weight"), out)
        assertTrue(!out.contains("layers[4]"), out)
    }

    @Test
    fun profileFlagSelectsTheDeviceRules() {
        assertEquals(PlannerProfile.MOBILE_2GB, profileFor("mobile"))
        assertEquals(PlannerProfile.DESKTOP, profileFor("desktop"))
        assertEquals(PlannerProfile.NATIVE, profileFor("native"))
        assertEquals(null, profileFor("none"), "no profile means the plan's own defaults, as before")
    }

    @Test
    fun theProfiledPlanPrintsItsRulesAboveTheTable() {
        // #1039: a plan printed months later has to say which rules produced it.
        val f = Format(FP32, TensorEncoding.Q4_K)
        val w = PlanTensor("blk.0.attn_q.weight", null, f, 600L * (1 shl 20) / 144 * 256, 600L shl 20)
        val g = ModelGeometry(16, 32, 8, 64, 64, 2048, 5632, 32_000)
        val input = PlanInput("m", "llama", listOf(w), g, ctx = 8192)

        val mobile = PlannerProfile.MOBILE_2GB.plan(input, availableBytes = 2048L shl 20).render()
        assertTrue(mobile.startsWith("profile mobile-2gb"), mobile)
        assertTrue(mobile.contains("prefill 256"), mobile)
        assertTrue(mobile.contains("note: KV cache switched"), "a tight mobile plan quantizes the cache:\n$mobile")

        val desktop = PlannerProfile.DESKTOP.plan(input, availableBytes = 2048L shl 20).render()
        assertTrue(desktop.startsWith("profile desktop"), desktop)
        assertTrue(!desktop.contains("switched"), "the desktop profile keeps today's behaviour:\n$desktop")
    }
}
