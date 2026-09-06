package sk.ainet.bench.publish.schema

import kotlinx.serialization.json.Json
import sk.ainet.lang.memory.trace.GenerationMetrics
import sk.ainet.lang.memory.trace.ModuleCost
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1035: the generation metrics survive the trip into the benchmark JSON that dashboards and the
 * Phoronix upload read, under the names `scripts/check_engine_json.sh` validates.
 *
 * The written fixture is the script's own test input: `./scripts/check_engine_json.sh
 * skainet-backends/benchmarks/jvm-cpu-publish/build/engine-json-check` must pass on it.
 */
class GenerationMetricsRecordTest {

    private val json = Json { prettyPrint = true; encodeDefaults = true; explicitNulls = false }

    private val metrics = GenerationMetrics(
        prefillTokens = 128,
        prefillNanos = 320_000_000L,
        decodeSteps = 64,
        decodeNanos = 1_280_000_000L,
        sampleNanos = 6_400_000L,
        timeToFirstTokenNanos = 340_000_000L,
        bytesReadDuringDecode = 40L * 1024 * 1024 * 1024,
        bytesWrittenDuringDecode = 4L * 1024 * 1024,
        kernelNanosDuringDecode = 1_024_000_000L,
        kernelRunsDuringDecode = 4_096,
        adapterCount = 2,
        adapterBytes = 1_048_576L,
        modules = listOf(ModuleCost("model.layers[0].attn", 400_000_000L, 64), ModuleCost("model.layers[0].mlp", 600_000_000L, 64)),
        pageFaultsDuringDecode = 3L,
        peakBytesPerSecond = 50L * 1024 * 1024 * 1024,
    )

    private fun record(generation: GenerationMetricsRecord?) = BenchmarkRecord(
        schemaVersion = "1.0.0",
        suite = "skainet-engine",
        scenario = "decode-synthetic",
        publishedAt = "2026-08-24T00:00:00Z",
        runtime = RuntimeInfo(
            version = "0.40.1", commit = "abcdef0", backend = "cpu",
            kernelProvider = "scalar", availableProviders = listOf("scalar"),
        ),
        system = SystemInfo(
            os = "linux", arch = "x86_64", cpu = "test", cpuLogicalCores = 8,
            memoryGib = 32L, jdk = "25", jdkVendor = "test",
        ),
        config = RunConfig(
            warmupRuns = 1, measuredRuns = 3, seed = 1L,
            parameters = mapOf("ctx" to "512"), jvmArgs = emptyList(), smokeMode = true,
        ),
        metrics = MetricSet("decode_tokens_per_second", "tok/s", 50.0, 0.5, 49.0, 51.0, 1.0),
        samples = listOf(49.0, 50.0, 51.0),
        generation = generation,
    )

    @Test
    fun `the metrics map onto the published field names`() {
        val rec = metrics.toRecord()
        assertEquals(128, rec.prefillTokens)
        assertEquals(64, rec.decodeSteps)
        assertEquals(50.0, rec.decodeTokensPerSecond!!, 1e-9, "64 steps in 1.28 s")
        assertEquals(340.0, rec.ttftMs!!, 1e-9)
        assertEquals(20.0, rec.msPerDecodeStep, 1e-9)
        assertEquals(62.5, rec.bandwidthUtilizationPercent!!, 1e-6, "31.25 GB/s of a 50 GB/s device")
        assertEquals(80.0, rec.kernelShareOfDecodePercent!!, 1e-9)
        assertEquals(2, rec.adapterCount)
        assertEquals(3L, rec.pageFaults)
        assertEquals(setOf("model.layers[0].attn", "model.layers[0].mlp"), rec.moduleBreakdownMs.keys)
        assertEquals(600.0, rec.moduleBreakdownMs.getValue("model.layers[0].mlp"), 1e-9)
    }

    @Test
    fun `a record with generation metrics serializes under the names the checker requires`() {
        val text = json.encodeToString(BenchmarkRecord.serializer(), record(metrics.toRecord()))
        for (key in listOf(
            "\"generation\"", "\"prefill_tokens\"", "\"decode_steps\"", "\"ms_per_decode_step\"",
            "\"bytes_read\"", "\"adapter_count\"", "\"adapter_bytes\"",
            "\"decode_tokens_per_second\"", "\"effective_bandwidth_bytes_per_second\"",
            "\"bandwidth_utilization_percent\"", "\"module_breakdown_ms\"", "\"ttft_ms\"",
        )) {
            assertTrue(text.contains(key), "missing $key in:\n$text")
        }
        val dir = File("build/engine-json-check").apply { mkdirs() }
        File(dir, "decode-synthetic.json").writeText(text)
    }

    @Test
    fun `a scenario without a generation loop omits the block entirely`() {
        val text = json.encodeToString(BenchmarkRecord.serializer(), record(null))
        assertFalse(text.contains("\"generation\""), "a matmul scenario must not carry an empty generation block")
        File("build/engine-json-check").apply { mkdirs() }
        File("build/engine-json-check/matmul-only.json").writeText(text)
    }
}
