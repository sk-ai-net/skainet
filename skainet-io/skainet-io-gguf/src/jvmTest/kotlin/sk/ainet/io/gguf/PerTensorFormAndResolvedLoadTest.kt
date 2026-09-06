package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.KernelCapabilities
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.memory.plan.StorageCapabilities
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * #1144: plan → load wiring. A per-tensor form function outranks the uniform form; `ResolvedGguf`
 * feeds the loader what the resolvers decided; the user's override outranks the resolver; and the
 * decisions are priceable and explainable before the payload is read.
 */
class PerTensorFormAndResolvedLoadTest {

    private fun file(): File = SyntheticGguf.write(
        SyntheticGguf.tensor("w_f32", GGMLQuantizationType.F32, elements = 1024),
        SyntheticGguf.tensor("w_q4k", GGMLQuantizationType.Q4_K, elements = 1024),
        SyntheticGguf.tensor("w_q80", GGMLQuantizationType.Q8_0, elements = 768)
            .copy(dims = listOf(256L, 3L)),
    )

    private fun loadVia(f: File, build: (() -> JvmRandomAccessSource) -> StreamingGgufParametersLoader):
        Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking {
            build { JvmRandomAccessSource.open(f) }
                .load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    @Test
    fun `a uniform weightFormFor is bit-identical to the single form`() {
        val f = file()
        try {
            for (form in listOf(
                WeightForm.AS_STORED_ON_HEAP,
                WeightForm(residency = WeightResidency.MAPPED),
                WeightForm(encoding = EncodingRequest.DequantizeTo(FP32)),
            )) {
                val uniform = loadVia(f) { src ->
                    StreamingGgufParametersLoader(sourceProvider = src, weightForm = form)
                }
                val perTensor = loadVia(f) { src ->
                    StreamingGgufParametersLoader(sourceProvider = src, weightFormFor = { form })
                }
                assertEquals(uniform.keys, perTensor.keys, "$form: different tensors came out")
                for ((name, u) in uniform) {
                    assertContentEquals(
                        u.data.copyToFloatArray(),
                        perTensor.getValue(name).data.copyToFloatArray(),
                        "$form: $name values",
                    )
                    assertEquals(u.shape, perTensor.getValue(name).shape, "$form: $name shape")
                }
            }
        } finally {
            f.delete()
        }
    }

    @Test
    fun `per-tensor forms are honoured per tensor`() {
        val f = file()
        try {
            val loaded = loadVia(f) { src ->
                StreamingGgufParametersLoader(
                    sourceProvider = src,
                    weightFormFor = { name ->
                        when (name) {
                            "w_q4k" -> WeightForm(encoding = EncodingRequest.DequantizeTo(FP32))
                            else -> null // uniform default: as stored, on heap
                        }
                    },
                )
            }
            assertTrue(loaded.getValue("w_q4k").data !is PackedBlockStorage, "w_q4k should be dense")
            assertTrue(loaded.getValue("w_q80").data is PackedBlockStorage, "w_q80 should stay packed")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `ResolvedGguf resolves and the loader obeys`() {
        val f = file()
        try {
            // DENSE_ONLY kernels: the resolver must dequantize every packed weight
            val resolution = ResolvedGguf.resolve(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                profile = PlannerProfile.DESKTOP,
                capabilities = KernelCapabilities.DENSE_ONLY,
                platform = StorageCapabilities.FULL,
            )
            val q4k = resolution.input.weights.first { it.name == "w_q4k" }
            assertTrue(q4k.form?.encoding is EncodingRequest.DequantizeTo, "resolver should dequantize q4k")
            assertTrue(q4k.residentBytes > q4k.bytes, "dense costs more than packed and the plan must say so")

            val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
            runBlocking {
                resolution.loader.load<FP32, Float>(DefaultDataExecutionContext(), FP32::class) { name, tensor ->
                    loaded[name] = tensor
                }
            }
            assertTrue(loaded.getValue("w_q4k").data !is PackedBlockStorage, "loader must obey the resolved form")

            val explains = resolution.explainPlacements()
            assertEquals(resolution.input.weights.size, explains.size)
            assertTrue(explains.any { "w_q4k" in it }, explains.joinToString("\n"))

            val plan = resolution.profiledPlan(availableBytes = 8L * 1024 * 1024 * 1024)
            assertEquals(resolution.input.weights.sumOf { it.residentBytes }, plan.plan.weightsBytes)
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the override outranks the resolver`() {
        val f = file()
        try {
            val resolution = ResolvedGguf.resolve(
                sourceProvider = { JvmRandomAccessSource.open(f) },
                profile = PlannerProfile.DESKTOP,
                capabilities = KernelCapabilities.DENSE_ONLY,
                platform = StorageCapabilities.FULL,
                overrideFormFor = { name -> if (name == "w_q4k") WeightForm.AS_STORED_ON_HEAP else null },
            )
            val q4k = resolution.input.weights.first { it.name == "w_q4k" }
            assertEquals(EncodingRequest.KeepAsStored, q4k.form?.encoding, "override must win")

            val loaded = LinkedHashMap<String, Tensor<FP32, Float>>()
            runBlocking {
                resolution.loader.load<FP32, Float>(DefaultDataExecutionContext(), FP32::class) { name, tensor ->
                    loaded[name] = tensor
                }
            }
            assertTrue(loaded.getValue("w_q4k").data is PackedBlockStorage, "override said keep packed")
            assertTrue(loaded.getValue("w_q80").data !is PackedBlockStorage, "un-overridden tensors follow the resolver")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `a strict profile refuses at resolve time — before any payload is read`() {
        val f = file()
        try {
            assertFailsWith<IllegalStateException> {
                ResolvedGguf.resolve(
                    sourceProvider = { JvmRandomAccessSource.open(f) },
                    profile = PlannerProfile.MOBILE_2GB, // strict
                    capabilities = KernelCapabilities.DENSE_ONLY,
                    platform = StorageCapabilities.FULL,
                )
            }
        } finally {
            f.delete()
        }
    }
}
