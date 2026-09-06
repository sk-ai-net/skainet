package sk.ainet.bench.publish.schema

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import sk.ainet.lang.memory.trace.GenerationMetrics

/**
 * The generation-loop half of a benchmark record (SKEEP-003 §4.9, #1035): what a decode run
 * reported about itself, in the same JSON the dashboards and the Phoronix upload already read.
 *
 * Optional on [BenchmarkRecord] — a matmul microbenchmark has no generation loop and omits it —
 * and validated by `scripts/check_engine_json.sh` whenever it is present. Rates are nullable for
 * the same reason they are nullable on [GenerationMetrics]: a span too short for the platform
 * clock produces no rate rather than an infinite one.
 */
@Serializable
public data class GenerationMetricsRecord(
    @SerialName("prefill_tokens")
    val prefillTokens: Int,
    @SerialName("prefill_tokens_per_second")
    val prefillTokensPerSecond: Double? = null,
    @SerialName("decode_steps")
    val decodeSteps: Int,
    @SerialName("decode_tokens_per_second")
    val decodeTokensPerSecond: Double? = null,
    @SerialName("ttft_ms")
    val ttftMs: Double? = null,
    @SerialName("ms_per_decode_step")
    val msPerDecodeStep: Double,
    @SerialName("bytes_read")
    val bytesRead: Long,
    @SerialName("effective_bandwidth_bytes_per_second")
    val effectiveBandwidthBytesPerSecond: Double? = null,
    @SerialName("bandwidth_utilization_percent")
    val bandwidthUtilizationPercent: Double? = null,
    @SerialName("kernel_share_of_decode_percent")
    val kernelShareOfDecodePercent: Double? = null,
    @SerialName("adapter_count")
    val adapterCount: Int,
    @SerialName("adapter_bytes")
    val adapterBytes: Long,
    @SerialName("page_faults")
    val pageFaults: Long? = null,
    @SerialName("page_faults_per_second")
    val pageFaultsPerSecond: Double? = null,
    @SerialName("module_breakdown_ms")
    val moduleBreakdownMs: Map<String, Double> = emptyMap(),
)

/** This run's metrics as the record the benchmark JSON carries. */
public fun GenerationMetrics.toRecord(): GenerationMetricsRecord = GenerationMetricsRecord(
    prefillTokens = prefillTokens,
    prefillTokensPerSecond = prefillTokensPerSecond,
    decodeSteps = decodeSteps,
    decodeTokensPerSecond = decodeTokensPerSecond,
    ttftMs = timeToFirstTokenNanos?.let { it / 1_000_000.0 },
    msPerDecodeStep = nanosPerDecodeStep / 1_000_000.0,
    bytesRead = bytesReadDuringDecode,
    effectiveBandwidthBytesPerSecond = effectiveBandwidthBytesPerSecond,
    bandwidthUtilizationPercent = bandwidthUtilization?.let { it * 100.0 },
    kernelShareOfDecodePercent = kernelShareOfDecode?.let { it * 100.0 },
    adapterCount = adapterCount,
    adapterBytes = adapterBytes,
    pageFaults = pageFaultsDuringDecode,
    pageFaultsPerSecond = pageFaultsPerSecond,
    moduleBreakdownMs = modules.associate { it.path to it.nanos / 1_000_000.0 },
)
