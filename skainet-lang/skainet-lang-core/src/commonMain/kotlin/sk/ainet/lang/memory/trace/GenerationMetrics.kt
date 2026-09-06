package sk.ainet.lang.memory.trace


/** Time spent inside one module span, and how often it ran (the per-layer breakdown). */
public data class ModuleCost(val path: String, val nanos: Long, val calls: Int) {
    public val averageNanos: Long get() = if (calls == 0) 0L else nanos / calls
}

/**
 * The generation-loop metrics of SKEEP-003 §4.9, derived from a recorded [TraceEvent] stream:
 * TTFT, prefill and decode tok/s, the per-module breakdown, what the adapters cost, the
 * **effective memory bandwidth** (bytes a decode step actually read ÷ how long it took) and the
 * page-fault rate that says whether mapped weights are being paged back in.
 *
 * Nothing here measures anything itself: the numbers come from the spans the loop opened
 * ([prefill], [decodeStep], [sample], [module]) and the events dispatch already emits, so a run
 * that is traced is a run that can be reported on. Rates are `null` rather than infinite when the
 * denominator is zero — a coarse clock (JS, Wasm) can legitimately time a short span as 0 ns.
 *
 * @property peakBytesPerSecond the device's peak memory bandwidth, if known; only then is
 *   [bandwidthUtilization] computable.
 */
public data class GenerationMetrics(
    val prefillTokens: Int = 0,
    val prefillNanos: Long = 0L,
    val decodeSteps: Int = 0,
    val decodeNanos: Long = 0L,
    val sampleNanos: Long = 0L,
    val timeToFirstTokenNanos: Long? = null,
    val bytesReadDuringDecode: Long = 0L,
    val bytesWrittenDuringDecode: Long = 0L,
    val kernelNanosDuringDecode: Long = 0L,
    val kernelRunsDuringDecode: Int = 0,
    val adapterCount: Int = 0,
    val adapterBytes: Long = 0L,
    val modules: List<ModuleCost> = emptyList(),
    val pageFaultsDuringDecode: Long? = null,
    val peakBytesPerSecond: Long? = null,
) {
    /** Prompt tokens per second, or `null` if the prefill was not timed. */
    public val prefillTokensPerSecond: Double?
        get() = perSecond(prefillTokens.toDouble(), prefillNanos)

    /** Decoded tokens per second, or `null` if no decode step was timed. */
    public val decodeTokensPerSecond: Double?
        get() = perSecond(decodeSteps.toDouble(), decodeNanos)

    /**
     * Bytes read per second across the decode phase — the number that says whether decode is
     * memory-bound, and the one a quantized format is supposed to improve.
     */
    public val effectiveBandwidthBytesPerSecond: Double?
        get() = perSecond(bytesReadDuringDecode.toDouble(), decodeNanos)

    /** [effectiveBandwidthBytesPerSecond] as a fraction of [peakBytesPerSecond] (1.0 = at peak). */
    public val bandwidthUtilization: Double?
        get() {
            val peak = peakBytesPerSecond ?: return null
            if (peak <= 0L) return null
            val effective = effectiveBandwidthBytesPerSecond ?: return null
            return effective / peak
        }

    /** Major page faults per second during decode — flat mapped weights mean this stays near zero. */
    public val pageFaultsPerSecond: Double?
        get() = pageFaultsDuringDecode?.let { perSecond(it.toDouble(), decodeNanos) }

    /** Bytes the dispatcher converted, as a fraction of the bytes decode read at all. */
    public val adapterShareOfBytesRead: Double?
        get() = if (bytesReadDuringDecode == 0L) null else adapterBytes.toDouble() / bytesReadDuringDecode

    /** Share of decode time spent inside kernels (the rest is dispatch, allocation, the loop itself). */
    public val kernelShareOfDecode: Double?
        get() = if (decodeNanos == 0L) null else kernelNanosDuringDecode.toDouble() / decodeNanos

    /** Average nanoseconds per decode step. */
    public val nanosPerDecodeStep: Long get() = if (decodeSteps == 0) 0L else decodeNanos / decodeSteps

    /** Emit the derived numbers as counters, so an exporter shows them beside the spans. */
    public fun emitTo(sink: TraceSink) {
        if (!sink.isEnabled) return
        timeToFirstTokenNanos?.let { sink.counter(Counters.TIME_TO_FIRST_TOKEN, it / 1_000, unit = "us") }
        prefillTokensPerSecond?.let { sink.counter(Counters.PREFILL_TOKENS_PER_SECOND, it.toLong(), unit = "tokens/s") }
        decodeTokensPerSecond?.let { sink.counter(Counters.DECODE_TOKENS_PER_SECOND, it.toLong(), unit = "tokens/s") }
        effectiveBandwidthBytesPerSecond?.let { sink.counter(Counters.EFFECTIVE_BANDWIDTH, it.toLong(), unit = "bytes/s") }
        bandwidthUtilization?.let { sink.counter(Counters.BANDWIDTH_UTILIZATION, (it * 100).toLong(), unit = "percent") }
    }

    /** A short human-readable table — what the decode sample prints and a PR quotes. */
    public fun render(): String = buildString {
        appendLine("generation metrics")
        appendLine("  prefill        ${prefillTokens} tokens in ${ms(prefillNanos)} ms${rate(prefillTokensPerSecond, "tok/s")}")
        appendLine("  decode         ${decodeSteps} steps in ${ms(decodeNanos)} ms${rate(decodeTokensPerSecond, "tok/s")}")
        appendLine("  ttft           ${timeToFirstTokenNanos?.let { ms(it) + " ms" } ?: "—"}")
        appendLine("  per step       ${ms(nanosPerDecodeStep)} ms")
        appendLine("  bytes read     $bytesReadDuringDecode in $kernelRunsDuringDecode kernel runs")
        appendLine("  bandwidth      ${effectiveBandwidthBytesPerSecond?.let { fmt(it / 1e9) + " GB/s" } ?: "—"}${
            bandwidthUtilization?.let { " (" + fmt(it * 100) + "% of peak)" } ?: ""
        }")
        appendLine("  adapters       $adapterCount, $adapterBytes bytes${adapterShareOfBytesRead?.let { " (" + fmt(it * 100) + "% of bytes read)" } ?: ""}")
        appendLine("  page faults    ${pageFaultsDuringDecode?.toString() ?: "—"}${rate(pageFaultsPerSecond, "/s")}")
        if (modules.isNotEmpty()) {
            appendLine("  modules")
            for (m in modules) appendLine("    ${m.path}  ${ms(m.nanos)} ms in ${m.calls} calls")
        }
    }

    private fun rate(value: Double?, unit: String): String = value?.let { " (${fmt(it)} $unit)" } ?: ""

    public companion object {
        private fun perSecond(count: Double, nanos: Long): Double? =
            if (nanos <= 0L) null else count * 1_000_000_000.0 / nanos

        private fun ms(nanos: Long): String = fmt(nanos / 1_000_000.0)

        private fun fmt(v: Double): String {
            val scaled = (v * 100).toLong()
            return "${scaled / 100}.${(scaled % 100).let { if (it < 0) -it else it }.toString().padStart(2, '0')}"
        }

        /**
         * Derive the metrics from [events] (a [RecordingTraceSink]'s stream).
         *
         * **TTFT** is measured from the start of the first `prefill` — or the first `decode` step
         * when a run has no prompt pass — to the end of the first `sample`, or of the first decode
         * step when the loop does not trace sampling.
         */
        public fun from(events: List<TraceEvent>, peakBytesPerSecond: Long? = null): GenerationMetrics {
            var prefillTokens = 0
            var prefillNanos = 0L
            var decodeNanos = 0L
            var sampleNanos = 0L
            val decodeSteps = HashSet<Int>()
            var untimedDecodeSteps = 0
            var bytesRead = 0L
            var bytesWritten = 0L
            var kernelNanos = 0L
            var kernelRuns = 0
            var adapterCount = 0
            var adapterBytes = 0L
            var firstPhaseStart: Long? = null
            var firstDecodeEnd: Long? = null
            var firstSampleEnd: Long? = null
            var pageFaultsFirst: Long? = null
            var pageFaultsLast: Long? = null
            val moduleNanos = LinkedHashMap<String, Long>()
            val moduleCalls = LinkedHashMap<String, Int>()

            // Open spans, innermost last. A KernelRun belongs to decode when a decode span is open.
            val open = ArrayList<TraceEvent.PhaseBegin>()
            fun inDecode(): Boolean = open.any { it.phase == Phases.DECODE }

            for (e in events) when (e) {
                is TraceEvent.PhaseBegin -> {
                    if (firstPhaseStart == null && (e.phase == Phases.PREFILL || e.phase == Phases.DECODE)) {
                        firstPhaseStart = e.timeNanos
                    }
                    if (e.phase == Phases.PREFILL) {
                        prefillTokens += e.attributes[Phases.ATTR_TOKENS]?.toIntOrNull() ?: 0
                    }
                    if (e.phase == Phases.DECODE) {
                        val step = e.step
                        if (step != null) decodeSteps += step else untimedDecodeSteps++
                    }
                    open.add(e)
                }
                is TraceEvent.PhaseEnd -> {
                    val idx = open.indexOfLast { it.phase == e.phase && it.step == e.step }
                    val begin = if (idx >= 0) open.removeAt(idx) else null
                    val duration = if (e.durationNanos > 0L) e.durationNanos else begin?.let { e.timeNanos - it.timeNanos } ?: 0L
                    when {
                        e.phase == Phases.PREFILL -> prefillNanos += duration
                        e.phase == Phases.DECODE -> decodeNanos += duration
                        e.phase == Phases.SAMPLE -> sampleNanos += duration
                        begin?.attributes?.get(Phases.ATTR_KIND) == Phases.KIND_MODULE -> {
                            moduleNanos[e.phase] = (moduleNanos[e.phase] ?: 0L) + duration
                            moduleCalls[e.phase] = (moduleCalls[e.phase] ?: 0) + 1
                        }
                    }
                    // the first token is out once the first sample closes — or, for a loop that does
                    // not trace sampling, once the first decode step does
                    if (firstSampleEnd == null && e.phase == Phases.SAMPLE) firstSampleEnd = e.timeNanos
                    if (firstDecodeEnd == null && e.phase == Phases.DECODE) firstDecodeEnd = e.timeNanos
                }
                is TraceEvent.KernelRun -> if (inDecode()) {
                    bytesRead += e.bytesRead
                    bytesWritten += e.bytesWritten
                    kernelNanos += e.durationNanos
                    kernelRuns++
                }
                is TraceEvent.AdapterInserted -> if (inDecode()) { adapterCount++; adapterBytes += e.bytes }
                is TraceEvent.Counter -> if (e.name == Counters.PAGE_FAULTS && inDecode()) {
                    if (pageFaultsFirst == null) pageFaultsFirst = e.value
                    pageFaultsLast = e.value
                }
                else -> Unit
            }

            val firstTokenEnd = firstSampleEnd ?: firstDecodeEnd
            val ttft = if (firstPhaseStart != null && firstTokenEnd != null) firstTokenEnd - firstPhaseStart else null
            val faults = if (pageFaultsFirst != null && pageFaultsLast != null) pageFaultsLast - pageFaultsFirst else null

            return GenerationMetrics(
                prefillTokens = prefillTokens,
                prefillNanos = prefillNanos,
                decodeSteps = decodeSteps.size + untimedDecodeSteps,
                decodeNanos = decodeNanos,
                sampleNanos = sampleNanos,
                timeToFirstTokenNanos = ttft,
                bytesReadDuringDecode = bytesRead,
                bytesWrittenDuringDecode = bytesWritten,
                kernelNanosDuringDecode = kernelNanos,
                kernelRunsDuringDecode = kernelRuns,
                adapterCount = adapterCount,
                adapterBytes = adapterBytes,
                modules = moduleNanos.map { (path, nanos) -> ModuleCost(path, nanos, moduleCalls[path] ?: 0) }
                    .sortedByDescending { it.nanos },
                pageFaultsDuringDecode = faults,
                peakBytesPerSecond = peakBytesPerSecond,
            )
        }

        /** Derive the metrics from what a [RecordingTraceSink] kept. */
        public fun from(sink: RecordingTraceSink, peakBytesPerSecond: Long? = null): GenerationMetrics =
            from(sink.events(), peakBytesPerSecond)
    }
}
