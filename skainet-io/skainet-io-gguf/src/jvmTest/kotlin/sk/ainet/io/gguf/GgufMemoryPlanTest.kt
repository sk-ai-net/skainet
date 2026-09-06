package sk.ainet.io.gguf

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** SKEEP-003 M0-F1: the plan comes from the GGUF header alone — tensor table + metadata, no tensor bytes. */
class GgufMemoryPlanTest {

    @Test
    fun planFromSyntheticHeaderOnly() {
        val q4 = SyntheticGguf.tensor("w.q4k", GGMLQuantizationType.Q4_K, 512)
        val q8 = SyntheticGguf.tensor("w.q8", GGMLQuantizationType.Q8_0, 64)
        val f32 = SyntheticGguf.tensor("w.f32", GGMLQuantizationType.F32, 10)
        val file = SyntheticGguf.write(q4, q8, f32)
        JvmRandomAccessSource.open(file).use { src ->
            val reader = StreamingGGUFReader.open(src)
            val input = reader.planInput(ctx = 256)
            assertEquals("test", input.architecture)
            assertNull(input.geometry) // synthetic header carries no architecture geometry
            assertEquals(3, input.weights.size)
            val byName = input.weights.associateBy { it.name }
            assertEquals(FP32, byName["w.q4k"]!!.format.dtype); assertEquals(TensorEncoding.Q4_K, byName["w.q4k"]!!.format.encoding)
            assertEquals(144L * 2, byName["w.q4k"]!!.bytes)
            assertEquals(34L * 2, byName["w.q8"]!!.bytes)
            assertEquals(TensorEncoding.Dense(4), byName["w.f32"]!!.format.encoding); assertEquals(40L, byName["w.f32"]!!.bytes)
            // unknown architecture: no name map, names kept as unmapped
            assertEquals(listOf("w.q4k", "w.q8", "w.f32"), input.unmappedWeights)
            val plan = MemoryPlans.plan(input, Budget.of(1L shl 30))
            assertEquals(288L + 68 + 40, plan.weightsBytes)
            assertEquals(true, plan.fits)
        }
    }

    @Test
    fun ggufFormatMapping() {
        assertEquals(TensorEncoding.Q6_K, ggufFormat(GGMLQuantizationType.Q6_K, 0).encoding)
        assertEquals(FP32, ggufFormat(GGMLQuantizationType.Q6_K, 0).dtype)
        assertEquals(TensorEncoding.Dense(2), ggufFormat(GGMLQuantizationType.BF16, 0).encoding)
        // #1033: the ternary types are described, not opaque — a TQ1_0 tensor plans at 54 B per
        // 256 elements instead of falling back to "whatever the header said the bytes were".
        assertEquals(TensorEncoding.TQ1_0, ggufFormat(GGMLQuantizationType.TQ1_0, 123).encoding)
        assertEquals(TensorEncoding.TQ2_0, ggufFormat(GGMLQuantizationType.TQ2_0, 123).encoding)
        assertEquals(FP32, ggufFormat(GGMLQuantizationType.TQ1_0, 123).dtype)
        assertEquals(54L, ggufFormat(GGMLQuantizationType.TQ1_0, 0).physicalBytes(256))
        assertEquals(66L, ggufFormat(GGMLQuantizationType.TQ2_0, 0).physicalBytes(256))
        assertTrue(ggufFormat(GGMLQuantizationType.IQ2_XS, 123).encoding is TensorEncoding.Opaque, "still unmapped")
    }

    /** Real file, fixture-gated (see GgufNameMapFixtureTest): the plan must be consistent with the header. */
    @Test
    fun planFromQwenFixture() {
        val dir = File(System.getProperty("skainet.test.fixturesDir") ?: "../skainet-io-core/build/test-fixtures")
        val f = File(dir, "Qwen2.5-0.5B-Instruct-Q8_0.gguf")
        if (!f.isFile) { println("[skip] ${f.name} not present"); return }
        JvmRandomAccessSource.open(f).use { src ->
            val reader = StreamingGGUFReader.open(src)
            val input = reader.planInput(ctx = 2048)
            val g = assertNotNull(input.geometry)
            assertEquals(24, g.layers); assertEquals(896, g.embeddingLength); assertEquals(14, g.heads); assertEquals(2, g.kvHeads)
            assertTrue(input.unmappedWeights.isEmpty(), "unmapped: ${input.unmappedWeights}")
            val plan = MemoryPlans.plan(input, Budget.of(1300L shl 20))
            // Q8_0 0.5B: weights ≈ file size (header excluded), within 2 %
            val packedBytes = reader.tensors.sumOf { it.nBytes }
            assertTrue(kotlin.math.abs(plan.weightsBytes - packedBytes) <= packedBytes / 50, "weights ${plan.weightsBytes} vs packed $packedBytes")
            println(plan.render())
        }
    }
}
