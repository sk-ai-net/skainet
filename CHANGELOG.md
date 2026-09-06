# Changelog

## [Unreleased]

### Added

- **Schedules — the compute-level algorithm/schedule split (SKEEP-005)**
  ([#1259](https://github.com/SKaiNET-developers/SKaiNET/issues/1259) registries not safe for concurrent reads,
  [#1260](https://github.com/SKaiNET-developers/SKaiNET/issues/1260) `parallelChunks` `runBlocking` island): a dependency-free
  `sk.ainet.context.schedule.Schedule` on every `ExecutionContext` (`ctx.schedule`,
  `ctx.withSchedule(s) { … }`, `ScheduledExecutionContext`), a JVM `CoroutineSchedule` built on
  structured concurrency (caller runs the first chunk, nested regions inline, `dedicated()` pool),
  and the first scheduled op — `scaledDotProductAttention` runs its `(batch, head)` units on the
  context's schedule, bit-identical to the sequential loop. `parallelChunks` no longer hides a
  `runBlocking(Dispatchers.Default)` island: it delegates to the ops' schedule, so
  `DirectCpuExecutionContext(schedule = Schedule.Sequential)` makes every kernel single-threaded
  and the JVM default (`CoroutineSchedule.hardware()`) spreads them across cores. Unhonoured
  requests are visible as `TraceEvent.ScheduleDowngraded`; regions as `TraceEvent.ScheduleRegion`.
  Compile lane: `dag { schedule(parallel("heads")) { … } }` stamps a `ScheduleHint` that
  `ScheduleAnnotationPass` validates per op and `StableHloConverter` emits as the
  `skainet.schedule` module attribute beside `skainet.tensor_layouts`. Registries
  (`KernelDispatch`, `KernelRegistry`) are now safe for concurrent reads. Found on the way and
  reported, not fixed here: [#1261](https://github.com/SKaiNET-developers/SKaiNET/issues/1261)
  (Panama matmul's first call differs by an ULP until the JIT intrinsifies `reduceLanes`). Docs: SKEEP-005,
  "Algorithm and schedule", "Schedule getting started" (executable sample).
- **`tensorFilter` on the single-file `SafeTensorsParametersLoader`**
  ([#1256](https://github.com/SKaiNET-developers/SKaiNET/issues/1256)): parity with the sharded
  loader — an optional predicate over the tensor headers; filtered-out tensors are neither read
  nor delivered and do not count toward progress. Lets a family load selectively from a
  checkpoint that also carries tensors the requested dtype cannot accept (int64 index tables,
  custom quantized payloads) instead of failing on the first unmapped one. Threaded through
  `withPolicy`.

### Changed

- **`ExperimentalMemoryApi` opt-in gate removed** (SKEEP-003): the `sk.ainet.lang.memory` API
  (`Storage`, `Scope`, `Format`, `Layout`, `TensorView`, `WeightForm`, `WeightByteOrder`, …) no
  longer requires `@OptIn(ExperimentalMemoryApi::class)` to use. The annotation's own message —
  "usable, but may change until milestone M1 is complete" — was stale: M0/M1/M2 all shipped
  complete in 0.49.0. `ExperimentalMemoryApi` is deleted along with every `@OptIn`/
  `@ExperimentalMemoryApi` annotation referencing it.

## [0.53.0] - 2026-09-02

Headline: **the export pipeline emits billion-parameter models.** Tracing a 4.5B-parameter
Gemma 3n E2B through the DSL → tape → StableHLO path could not finish on a 48 GB host
([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): shape-only tracing
materialized real zero buffers, constant extraction copied every weight while the originals
stayed live, and when the converter did fail it said so in MLIR comments and exited 0 — so the
first broken node cascaded through a thousand more and still produced a "module". Each of those
is closed, and the last one turned out to hide a fourth: the tied embedding is exactly one byte
larger than a JVM array can hold, which no amount of widening could fix — external constants now
travel as the aliased float array they already are. The same repro that OOMed a 46 GB heap now
exports the full model in under a minute with zero failure comments, and the sharded SafeTensors
loader the transformers families kept re-implementing lives in the engine.

### Added

- **`ShardedSafeTensorsParametersLoader` — the sharded-index SafeTensors loader**
  ([#1246](https://github.com/SKaiNET-developers/SKaiNET/issues/1246),
  [#1252](https://github.com/SKaiNET-developers/SKaiNET/pull/1252)): a `ParametersLoader` over
  `model.safetensors.index.json`, riding `StreamingShardedSafeTensorsReader.openFromIndex` with the
  single-file loader's BF16/FP16 policies (`withPolicy` parity), a fail-fast dtype pre-scan before
  any tensor is delivered, and a `tensorFilter` hook so size guards and name allowlists stay
  family-side while every dtype decision stays in the engine. The dtype dispatch and dequant
  helpers moved into a shared internal `SafeTensorsMaterializer`, so both loaders materialize
  identically (parity-tested). Downstream: SKaiNET-transformers' hand-rolled per-family SafeTensors
  loaders can collapse onto it.
- **`BufferHandle.Floats` — array-free path for ≥2 GiB constants**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): external FP32 constants
  now ride the aliased `FloatArray` end-to-end (graph → `ExternalParameterRef` → `.irpa`), never
  serializing to a single `ByteArray` — the gemma3n tied embedding (262144x2048 FP32 =
  `Int.MAX_VALUE` + 1 bytes) structurally cannot exist as one byte buffer. `IrpaWriter` streams
  the values little-endian in 64 MiB chunks; `DefaultBufferResolver` reads through a chunked byte
  view. Constant element counts fold in `Long`, and an oversized single-buffer serialization now
  throws `ConstantTooLargeException` with the remediation instead of the
  `NegativeArraySizeException` that was previously mistaken for a registry miss.

### Changed

- **Void tracing allocates nothing**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247),
  [#1249](https://github.com/SKaiNET-developers/SKaiNET/pull/1249)): `VoidTensorOps` recorded every
  shape-propagation op into a real dense zero buffer, allocated twice and retained by the trace —
  ~9 GB of zeros across a 30-layer Gemma 3n E2B trace. Static-shape void results are now lazy
  placeholders (readers still see zeros; unread ops allocate nothing), `ShapeOnlyTensorData` is
  public for consumers that hand-roll shape-only data, and `matmulWeightTransposed` no longer
  constructs the transposed intermediate. A 4096×4096 traced op stack tracks 0 bytes.
- **Graph constants alias live weights; packed params fail loudly**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): `TraceToGraphBuilder` no
  longer copies every frozen float weight into the graph — the constant's `initial_value` aliases
  the live buffer (read-only contract), halving weight residency during export. BF16/FP16 dense
  weights widen to one FP32 copy. A frozen parameter with packed storage (Q4_K, Q8_0, ternary, …)
  now throws `PackedConstantException` instead of silently becoming a function argument and
  producing an unservable module; `PackedConstantHandling.DEQUANTIZE` (threaded through
  `toComputeGraph`) opts into dense FP32 extraction instead.
- **StableHLO conversion fails loudly by default**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): converter failures were
  MLIR comments — a graph whose first node failed to lower could cascade through every downstream
  node and still "succeed" with an empty `return` and exit 0. `StableHloConverter` and every
  `StableHloConverterFactory` entry point now take a `ConversionErrorPolicy` (default `STRICT`):
  an unconvertible node throws `HloConversionException`, and an operand whose producer was never
  converted throws `MissingOperandException` instead of being silently dropped and shifting later
  operands into earlier positions. `ConversionErrorPolicy.LENIENT` restores the historical
  comment-and-continue behavior for callers that inspect partially-converted modules.

### Fixed

- **`clamp` lowers to StableHLO**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): the traced
  `clamp(x, minVal, maxVal)` op had no converter at all — the first gap the strict gemma3n export
  surfaced once failures stopped being comments. Lowers to `stablehlo.clamp` with splat bounds.
- **`indexSelect` is now routable in the StableHLO gather converter**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): the KSP tracing wrapper
  emits `indexSelect`, but the registry only knew `index_select`, so traced index-select nodes
  (e.g. Gemma per-layer embeddings) could not lower. Both spellings route to the gather lowering.
- **`StableHloConverterFactory.createBasic` registers `NeuralNetOperationsConverter`**
  ([#1247](https://github.com/SKaiNET-developers/SKaiNET/issues/1247)): parity with
  `createExtended` — a traced model with conv/pool/norm nodes no longer fails to lower via the
  basic factory, with registration order preserving existing op-name precedence.

## [0.52.0] - 2026-09-01

Headline: **the engine stops silently running on the scalar floor.** A downstream Gemma 4 port
was generating garbage at roughly 0.04 tok/s, and the investigation
([#1220](https://github.com/SKaiNET-developers/SKaiNET/issues/1220)) found the cause split across
both repositories — but the engine's share of it was one theme repeated: a fast path that exists,
is compiled in, and never gets used, with nothing saying so. `KernelDispatch` was never populated
in production at all, so every matmul fell back to the decoding reference kernel; dense FP32
weights in mapped or off-heap storage missed the kernel that serves them and dequantized instead;
and the fallback itself was routed to a no-op trace sink, which is why a ~1000x degradation could
sit in a release undetected. Those are closed, and the dispatcher now installs itself on first use
rather than trusting every entry point to remember — ternary/BitNet packs included,
so the discovery set covers every format the backends ship kernels for. Alongside that, Android's native targets now
run the whole dependency chain, not just its first two modules.

### Added

- **`ViewKernelPack` SPI and self-healing dispatch**
  ([#1220](https://github.com/SKaiNET-developers/SKaiNET/issues/1220),
  [#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)): `KernelDispatch` populates
  itself on first use via `ensureInstalled()`, backed by a new `ViewKernelPack` service interface
  with `installPlatformKernelPacks()` actuals per platform — ServiceLoader-based on JVM and
  Android, explicit on Kotlin/Native, which has no ServiceLoader. Applications no longer have to
  call an install routine at startup and no longer silently lose every kernel when they forget.
  The two backends ship discovery metadata: `FfmRowMajorKernelPackFactory`
  (`skainet-backend-native-cpu`) and `JniMappedKernelPackFactory` (`skainet-backend-jni-cpu`).
- **`androidNativeArm32`/`androidNativeArm64` across the downstream chain**
  ([#1239](https://github.com/SKaiNET-developers/SKaiNET/pull/1239)): `skainet-io-gguf`,
  `skainet-lang-dag`, `skainet-compile-dag`, `skainet-compile-opt`, `skainet-backend-api`,
  `skainet-backend-cpu`, plus `skainet-lang-models` and `skainet-compile-json` to close the target
  set over test compilations. Only `skainet-io-core` and friends had these targets before, so a
  consumer building for an Android device could not resolve the rest of what it needed.
  `skainet-io-core`'s 64-bit split source set has no counterpart here: none of these modules has
  posix-typed code, so arm32's `Int`-width `ssize_t`/`size_t` does not reach them.
- **Mapped-serving encodings derived from kernel registrations**
  ([#1193](https://github.com/SKaiNET-developers/SKaiNET/issues/1193),
  [#1215](https://github.com/SKaiNET-developers/SKaiNET/pull/1215)):
  `KernelDispatch.mappedServableEncodings()` reports which encodings a `MappedCapableKernel`
  actually serves right now, replacing a hand-kept list that could drift from the registry it
  described.
- **`gemma4` in the model registries**
  ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)): `TokenizerFactory` accepts
  the architecture and `ModelArchitecture.ggufIdMap` maps `"gemma4"` to `GEMMA`, so a Gemma 4 GGUF
  loads through the engine's own routes instead of throwing.
- **Dense FP32 GEMV path in the Panama kernel**
  ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)): `PanamaVectorMatmulKernel`
  gains `gemvRows()` for the m ≤ 8 shapes a decode step actually issues — 16.7x at m=1 over the
  general blocked path, which was written for prefill-sized work.

### Fixed

- **Ternary kernel packs join the self-healing dispatch SPI**
  ([#1240](https://github.com/SKaiNET-developers/SKaiNET/issues/1240),
  [#1241](https://github.com/SKaiNET-developers/SKaiNET/pull/1241)): the `ServiceLoader`
  service files now list `FfmTernaryKernelPackFactory` (JVM jar) and
  `JniTernaryKernelPackFactory` (Android AAR), so `KernelDispatch.ensureInstalled()` wires the
  exact FP32×`BITNET_B1_58` LUT gemv and the fused `BITNET_PLANES` lm_head with no bootstrap
  call — without this, a consumer loading ternary weights silently got the int8-requantize or
  decoding-reference path (~120× slower per the #1141 bench) unless it called
  `NativeTernaryF32GemvKernel.install()` / `NativeTernaryLmheadKernel.install()` explicitly:
  the exact failure mode this release's self-healing dispatch exists to eliminate, closed for
  the ternary formats in the same release. Kotlin/Native still installs explicitly (no
  `ServiceLoader` there); the ternary tutorial's install table says which targets are automatic.

- **Dense FP32 weights in mapped or off-heap storage fell back to dequantization**
  ([#1218](https://github.com/SKaiNET-developers/SKaiNET/pull/1218)): the kernel that serves them
  only recognised `Heap`, so a memory-mapped model — the whole point of mapped staging — took the
  slow path. Now served from any storage kind.
- **The reference-kernel fallback was invisible**
  ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)): `DefaultCpuOps` hardcoded
  `NoopTraceSink` at both dispatch sites, so falling back to the decoding reference kernel emitted
  nothing. `KernelDispatch` gains a `defaultSink` and warns once, loudly, the first time it
  happens.
- **`SpecialTokenSplitter` lost word boundaries when decoding token by token**
  ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)): it did not override
  `decodeToken`, so streaming consumers of any SentencePiece GGUF with special tokens saw spaces
  disappear from the output. Also fixes `token_type` parsing, which discarded `UInt`-typed GGUF
  metadata and could silently drop a model's special tokens.
- **`ar`/`ranlib` selection for the aarch64 cross build on macOS hosts**
  ([#1209](https://github.com/SKaiNET-developers/SKaiNET/pull/1209)): the build picked the host's
  Mach-O tools for an ELF target, producing archives the linker rejected.

### Performance

- **Small-shape FP32 matmul** ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)):
  a direct-loop path under `SMALL_FP32_MATMUL_WORK` skips blocking overhead that costs more than it
  saves at decode sizes, and `transposedDenseWeight()` caches the transpose instead of rebuilding
  it per call. Measured end to end on a downstream Gemma 4 port: ~2.3x on both decode and prefill.

### Docs

- Kernel-selection explanation page, covering the two registries and how a weight reaches a kernel
  ([#1221](https://github.com/SKaiNET-developers/SKaiNET/pull/1221)).
- Architecture reference gains its missing building blocks — kernel dispatch, ternary, AOT
  conversion ([#1216](https://github.com/SKaiNET-developers/SKaiNET/pull/1216)).
- DARC/SKEEP onboarding, issue taxonomy and an `F1Score` worked example for contributors
  ([#1238](https://github.com/SKaiNET-developers/SKaiNET/pull/1238)).
- `GITFLOW.adoc` reconciled with the `main` branch reset, documenting the release sequence actually
  used from 0.51.0 onward ([#1213](https://github.com/SKaiNET-developers/SKaiNET/pull/1213)).
- Why the `GROUP_128`/`GROUP_64` native decode kernel was closed
  ([#1205](https://github.com/SKaiNET-developers/SKaiNET/issues/1205),
  [#1214](https://github.com/SKaiNET-developers/SKaiNET/pull/1214)).

### Dependencies

- `github/codeql-action/upload-sarif` 4.37.8 → 4.37.9
  ([#1236](https://github.com/SKaiNET-developers/SKaiNET/pull/1236)).

## [0.51.0] - 2026-08-29

Headline: **ternary/BitNet weights join the memory-mapped weight story 0.50.0 started for every
other quant format.** 0.50.0 shipped mapped staging for every GGML block format but left ternary
(`BITNET_B1_58`) heap-staging, flagged then as needing the work tracked in
[#1198](https://github.com/SKaiNET-developers/SKaiNET/issues/1198). That work is done: off-heap
storage removes the Android ART heap-cap OOM risk a repacked ternary weight used to carry, and a
`SEQUENTIAL`-layout (NeoGPU-converted) GGUF now gets a true zero-copy mmap load with no repack at
all. A new AOT converter lets a build that owns its model pipeline pay that repack cost once,
offline, instead of on every load — its IREE-facing counterpart lives in a new home,
[SKaiNET-IREE-tools](https://github.com/SKaiNET-developers/SKaiNET-IREE-tools), on the
architectural grounds that compiled-target-specific conversion belongs beside the compiler
toolchain it targets, not inside core (see [#1207](https://github.com/SKaiNET-developers/SKaiNET/issues/1207)).

### Added

- **Off-heap storage for packed ternary/quantized weights**
  ([#1202](https://github.com/SKaiNET-developers/SKaiNET/issues/1202),
  [#1206](https://github.com/SKaiNET-developers/SKaiNET/pull/1206)): `Storage` gains
  `copyInto`/`copyFrom` bulk byte primitives implemented for every concrete storage kind (`Heap`,
  `SegmentStorage`, `MappedFileStorage`, `DirectBufferStorage`, `MappedBufferStorage`,
  `NativeMallocStorage`, `NativeMappedStorage`). `PackedBlockStorage` gains a `packedStorage`
  property (default: wraps `packedData` in `Storage.Heap`, so every existing quantized format is
  unaffected); `packedView` reads from it instead of always re-wrapping the raw array.
  `BitNetB158TensorData` gains a `Storage`-backed constructor/`fromStorage()` factory — the
  per-element accessors lazily snapshot off-heap storage only if actually touched, so inference
  through the GEMV kernels never materializes a heap copy. `StreamingGgufParametersLoader`
  allocates off-heap for repacked I2_S payloads at or above `PlannerProfile.OFF_HEAP_THRESHOLD`
  (256 KB) instead of a permanent `ByteArray`.
- **Zero-copy native gemv for off-heap ternary weights**
  ([#1202](https://github.com/SKaiNET-developers/SKaiNET/issues/1202),
  [#1206](https://github.com/SKaiNET-developers/SKaiNET/pull/1206)): `TernaryF32GemvNative` gains
  `gemvPackedStorage()`, letting the JVM/FFM face hand a `SegmentStorage`'s `MemorySegment`
  straight to the native downcall — the weight is never copied, not even once, where the
  previous `gemvPacked` path re-copied the *entire* weight matrix into a fresh arena on every
  row of every call, independent of storage kind. `NativeTernaryF32ViewKernel` now dispatches on
  the weight's storage kind instead of only accepting `Storage.Heap`, so an off-heap ternary
  weight keeps the fast NEON/FFM path instead of silently falling back to the slow reference
  kernel.
- **Zero-copy mmap for `SEQUENTIAL`-layout I2_S tensors**
  ([#1203](https://github.com/SKaiNET-developers/SKaiNET/issues/1203),
  [#1208](https://github.com/SKaiNET-developers/SKaiNET/pull/1208)): `I2sRepack.toSequentialPayload`
  no longer copies a `SEQUENTIAL` buffer that's already exactly the target payload (the common
  NeoGPU-converted case). More significantly, the loader's mmap-eligibility branch — previously
  keyed on a hardcoded encoding whitelist that didn't include `I2_S` — now maps a `SEQUENTIAL`
  I2_S tensor directly off the file whenever its trailing bytes are provably the real scale (no
  companion `<name>_scale` tensor overriding them), giving it the same zero-copy path Q4_K/Q8_0/etc.
  already had. `GROUP_128`/`GROUP_64` payloads and companion-scored `SEQUENTIAL` files correctly
  keep repacking.
- **`I2sAotConverter`: AOT GGUF → GGUF conversion for I2_S**
  ([#1207](https://github.com/SKaiNET-developers/SKaiNET/issues/1207),
  [#1210](https://github.com/SKaiNET-developers/SKaiNET/pull/1210)): reads an arbitrary GGUF,
  repacks I2_S tensors into `SEQUENTIAL`+trailer order ahead of time, drops the now-redundant
  companion scale tensor, and passes every other tensor and all KV metadata through unchanged —
  the converted file always takes the new zero-copy mmap path above, with no on-device cost at
  all. `GgufTensorEntry` (the writer's tensor-entry type) gains a `rawBytes` passthrough mode to
  support this without going through the element-indexed `TensorFlatten` path.
- **GGUF I2_S → `.irpa` conversion** (IREE-facing counterpart, in
  [SKaiNET-IREE-tools#1](https://github.com/SKaiNET-developers/SKaiNET-IREE-tools/pull/1), not
  this repo): a standalone Python tool producing an IREE parameter archive directly from a
  GGUF's ternary tensors, for `iree-compile --iree-opt-import-parameters=`.

### Fixed

- **Scoped dense-FP32 activations silently fell out of the quantized matmul chooser**
  ([#1211](https://github.com/SKaiNET-developers/SKaiNET/pull/1211)): `chooseQuantizedMatmul2D`
  accepted only `FloatArrayTensorData`/`MemorySegmentBackedData` activations; a
  `ScopedExecutionContext` forward's slab-backed `StorageFloatTensorData` fell through to
  `matmulGeneric`, whose per-element `get()` on a `Q8MemorySegmentTensorData` weight returns the
  raw quantization byte, not the value — silently wrong logits (the #993 class of bug). Any dense
  activation (`encoding == null`) is now accepted via its own offset-aware `copyToFloatArray()`.

## [0.50.0] - 2026-08-28

Headline: **model size on Android is now a page-cache question, not a heap question — and decode is
2.5× faster.** Quantized weights are served straight from the memory-mapped GGUF file (zero copies,
zero relayout), the packed matmul kernels thread across cores, and a 1.0 GB Qwen2.5-1.5B Q4_K_M —
which OOM'd at load under the 256 MB ART cap in 0.49.0 — loads in ~0.4 s with 566 KB of weight heap
and decodes at 61–66 ms/step with zero steady-state page faults (Pixel 8a, measured by the M2-A5
harness). The same file-order kernels serve heap-staged weights too, closing a measured 48,771 ms/step
silent-fallback trap for mixed-quant models — and fallbacks are never silent again.

### Added

- **Packed-tensor mapped staging** ([#1189](https://github.com/SKaiNET-developers/SKaiNET/issues/1189),
  [#1190](https://github.com/SKaiNET-developers/SKaiNET/pull/1190)): under
  `WeightForm(residency = MAPPED)` every GGML block format (Q4_K, Q6_K, Q5_K, Q8_0, Q4_0, Q5_0, Q5_1 —
  [#1192](https://github.com/SKaiNET-developers/SKaiNET/issues/1192)) is served from file-backed pages:
  `BufferPackedTensorData` over `MappedBufferStorage`/`DirectBufferStorage`, row-major (`_rm`) C kernels
  that read canonical GGUF file order (no prepack, no relayout copy), reached via JNI direct-buffer
  entries on Android and FFM `MemorySegment.ofBuffer` on the JVM
  ([#1191](https://github.com/SKaiNET-developers/SKaiNET/issues/1191)). Ternary (`BITNET_B1_58`) still
  heap-stages — its load-time repack needs the sidecar cache tracked in
  [#1198](https://github.com/SKaiNET-developers/SKaiNET/issues/1198).
- **Threaded packed matmuls** ([#1195](https://github.com/SKaiNET-developers/SKaiNET/issues/1195),
  [#1196](https://github.com/SKaiNET-developers/SKaiNET/pull/1196)): a spin-then-park worker pool with
  guided row-grains, engaged at `outputDim ≥ 512`. The design is measurement-driven and the failed
  variants are documented in the PR: per-call `pthread_create` cost 994 ms/step against deep-idle
  cores, and every *sleeping* pool lost to one pegged big core because sub-millisecond bursts never
  build scheduler utilization — the ~1 ms spin before parking (the same trick llama.cpp uses) is what
  unlocks big cores at full clocks. 153 → 61 ms/step on the 1.5B; results are bit-identical to
  single-threaded (disjoint row ranges, unchanged per-row accumulation order).
- **Storage-polymorphic row-major dispatch, and no silent fallbacks**
  ([#1193](https://github.com/SKaiNET-developers/SKaiNET/issues/1193),
  [#1197](https://github.com/SKaiNET-developers/SKaiNET/pull/1197)): one kernel per format serves
  `BLOCKED_ROW_MAJOR` weights from mapped, direct **or heap** storage — an un-prepacked heap canonical
  weight used to fall to the decoding reference kernel silently (measured: 48,771 ms/step on
  SmolLM2-135M, whose non-256-multiple dims made llama.cpp's quantizer emit mostly Q8_0; now 33 ms/step,
  the fastest configuration measured). `ViewKernel` gained a sink-aware `run` overload and every packed
  bridge announces a reference fallback as a `KernelRun` trace event with the reason; the M2-A5 harness
  prints the count.
- **The plan tells the truth about mapped weights**
  ([#1190](https://github.com/SKaiNET-developers/SKaiNET/pull/1190)): `MemoryPlan.budgetedBytes`
  charges mapped-servable weights against device RAM/page cache instead of the heap budget
  (`fits`, suggestions and `PlannerProfile`'s KV auto-quantization follow), rendered as its own
  `mapped (page cache, evictable — not heap)` line. `AllocationResolver.servesFromMapping` is the one
  predicate the resolver, the plan and the loader share, gated by
  `StorageCapabilities.mappedServableEncodings`, so they cannot tell different stories; `planInput`
  takes the `WeightForm` the load will use.
- **Kernel-support matrix: mapped serving section** — the generated matrix now shows, per platform,
  which formats serve from a mapping (all seven on Android `native-jni-direct` and JVM `ffm-rowmajor`;
  empty cells are the documented gaps).
- **"The DSL is compute"** ([#1194](https://github.com/SKaiNET-developers/SKaiNET/pull/1194)):
  the architecture principle behind all of the above as a teachable explanation page — network
  definitions describe computation only; memory intent lives in `WeightForm` at the load boundary,
  priced by the plan, honored-or-visibly-rejected by resolvers, with runtime dispatch holding no
  memory policy.

### Changed

- Cross-order kernel results (row-major vs feed-order) are **numerically equivalent, not bit-exact**:
  under `-O3 -ffast-math` compilers may contract float accumulation differently per loop shape
  (measured 2 ULP on clang/arm64, more under MSVC auto-vectorization). Integer-dot formats (Q4_K, Q6_K)
  currently match exactly, but only threaded-vs-single-threaded identity is contractual. The parity
  suites encode this.
- The M2-A5 measurement harness gained `residency=heap|mapped` and reports mapped vs heap weight bytes
  and the packed-kernel fallback count.

### Third-party

- The vendored NeoGPU ternary NEON kernel
  (`skainet-backends/skainet-backend-native-cpu/native/src/vendor/neogpu/hs_ml_ternary_neon.c`,
  © 2024 NeoGPU Contributors, MIT, byte-identical to upstream
  [anjaustin/neogpu](https://github.com/anjaustin/neogpu) @ `0846b24`) ships unchanged in this release;
  REUSE metadata and `META-INF/THIRD-PARTY-NOTICES.md` in the published artifacts carry the attribution
  ([#1166](https://github.com/SKaiNET-developers/SKaiNET/issues/1166)).

## [0.49.0] - 2026-08-26

Headline: **the SKEEP-003 memory & storage architecture, complete — from accepted proposal to shipped system.**
One storage model (`Storage` / `Scope` / `Format` / `Layout` / `TensorView`), resolver-owned decisions
(what form a weight takes, where its bytes live — never decided by the model author), scope-recycled eager
execution (flat-memory decode), and a compile lane that carries what the runtime decides into the exported
MLIR and `.irpa`. The version jump (0.40.1 → 0.49.0, ~100 merged PRs) is deliberate: this is the release
downstream repositories (SKaiNET-transformers, the IREE conformity pipeline) should build on, and it
removes every façade the architecture replaced. See **Breaking changes** below for the migration map.

### Breaking changes

- **The three legacy loader axes are gone** ([#1159](https://github.com/SKaiNET-developers/SKaiNET/issues/1159)):
  `QuantPolicy`, `StagingPolicy` and `WeightOrientation` are deleted. The loader takes one
  `WeightForm(encoding, order, shape, residency)` (uniform `weightForm` or per-tensor `weightFormFor`);
  migration: `DEQUANTIZE_TO_FP32` → `EncodingRequest.DequantizeTo(FP32)`, `StagingPolicy.MAPPED` →
  `WeightForm(residency = WeightResidency.MAPPED)`, `WeightOrientation.OUT_IN` → `WeightShapeOrientation.OUT_IN`.
  `AndroidGguf.loader` takes a `WeightForm` (default mapped residency).
- **The dead placement machinery is gone** ([#1142](https://github.com/SKaiNET-developers/SKaiNET/issues/1142)):
  `@Place`/`@Weights` (declared, retained, read by nothing), `sk.ainet.lang.tensor.storage.MemoryPlanner`,
  `StorageSpec`, and `Placement.residency`/`Residency`. Lifetime is `ScopeKind`; weight staging is
  `WeightForm.WeightResidency`; placement is decided by `AllocationResolver` (see Added). `Placement`
  itself stays (KV-cache stores carry it); `@KvCache`/`@KvCacheBypass` moved to `KvCacheAnnotations.kt`.
- **The `skainet.tensor_encodings` module attribute is gone**
  ([#1179](https://github.com/SKaiNET-developers/SKaiNET/issues/1179)): replaced by the machine-readable
  `skainet.tensor_layouts` (`{kind, block_elems, block_bytes, bits, block_order}`); nothing outside this
  repository read the old names dictionary.
- `LogicalDType` is deprecated end to end in favour of `DType`
  ([#1014](https://github.com/SKaiNET-developers/SKaiNET/issues/1014)); removal at the next major.

### Added

- **The memory model (SKEEP-003 M0 "know before you load" / M1 "flat decode" / M2 "1.58-bit on a 2 GB board")**
  ([#1001](https://github.com/SKaiNET-developers/SKaiNET/issues/1001),
  [#1002](https://github.com/SKaiNET-developers/SKaiNET/issues/1002),
  [#1003](https://github.com/SKaiNET-developers/SKaiNET/issues/1003); slices #1004–#1042):
  `Storage` (Heap/OffHeap/Mapped, ownership + liveness — a use-after-free is a loud
  `StorageClosedException`), `Scope` (`ModelScope`/`ForwardScope` slab with `reset()`),
  `Format(dtype, encoding)`, `Layout` (strides/offset/block geometry), `TensorView` with `prepack()` as
  the visible relayout and `materialize()` as the single copy point; `TraceSink` events (allocations,
  scope resets, adapter insertions); `MemoryPlan`/`MemoryPlans` header-only planning with budgets,
  suggestions and a plan-vs-actual check; the `skainet-plan` CLI; `PlannerProfile` (`MOBILE_2GB` with
  automatic KV quantization — and `strict`, so a missing kernel refuses instead of silently costing
  several times the weight, [#1128](https://github.com/SKaiNET-developers/SKaiNET/pull/1128));
  Android mmap loading + device fit checks; KV cache preallocated in model scope with declared formats;
  kernel dispatch on declared formats (`KernelKey`, registry-backed capabilities); the decode harness
  and M2 acceptance runs.
- **Weight forms — one resolved decision instead of three caller flags**
  ([#1109](https://github.com/SKaiNET-developers/SKaiNET/issues/1109) arc, #1114–#1120):
  `WeightForm` (encoding × byte order × shape × residency) resolved by `WeightFormResolver` from
  *what the file holds × the profile × what the backend's kernels can feed*; the loader honours it,
  the plan prices it (a resolved dequantization shows in the table, not at the OOM), conversions are
  traced, and packed weights can load directly in kernel-feed order
  ([#1120](https://github.com/SKaiNET-developers/SKaiNET/issues/1120)).
- **Placement is resolver-owned** ([#1133](https://github.com/SKaiNET-developers/SKaiNET/issues/1133) →
  #1142–#1144): `AllocationResolver.resolve(weight, profile, platform)` decides memory domain and scope
  (mapping requires: the form asks, the platform can, the bytes are the file's bytes);
  `AllocationResolver.explain()` renders every decision with its reason pre-load; `ResolvedGguf` wires
  plan → load with the documented user-wins precedence (per-tensor `weightFormFor` > uniform
  `weightForm` > resolver).
- **Scope-recycled eager execution** ([#1135](https://github.com/SKaiNET-developers/SKaiNET/issues/1135) →
  #1145/#1146/#1173): `ExecutionContext.memoryScope` is consulted by tensor creation *and* op outputs
  (`TensorDataFactory.adoptFloatArray`, `ScopedTensorDataFactory`), so
  `ctx.forwardScope(slabFloats) { … }` gives steady-state decode that allocates zero new slab bytes per
  step; the FP32 fast paths and the JVM Panama vector kernels are offset-aware, so slab-backed tensors
  keep SIMD speed.
- **Model-footprint analysis for GGUF, safetensors and ONNX**
  ([#1169](https://github.com/SKaiNET-developers/SKaiNET/issues/1169)): header-only `planInput` for all
  three formats (ONNX `external_data` sidecars priced correctly — multi-GB models no longer report ~0
  bytes; sizes `Long`-safe), and `PlannerProfile.EDGE` for embedded devices where the budget *is* the
  usable RAM. "Will it fit in ~2.1 GB?" is answered in seconds, without reading a tensor payload.
- **The compile lane carries what the runtime decides**
  ([#1147](https://github.com/SKaiNET-developers/SKaiNET/issues/1147) → #1178/#1179/#1180):
  `TensorRef` carries tensor identity (`TraceSession.identify`, registered from
  `trainableParameters()`), the encoding *object* (block size intact) and packed block order across the
  trace→graph boundary; the emitted module header declares structural facts per tensor
  (`skainet.tensor_layouts`); `ExternalParameterRef` declares block order to the `.irpa` consumer;
  `HloGenerator.generate(target = …)` runs the optimizer pipeline on the production path with
  `LayoutAssignmentPass` (rank-2 packed weights get kernel-feed order; the tape's carried facts are
  never overridden), and the `ResolvedComputeGraph` seams surface exactly the decisions made.
- **BitNet / ternary compute track** (#1033, #1040/#1041, #1136–#1141, #1150):
  ternary encodings (`TQ1_0`/`TQ2_0`, `BITNET_B1_58`, `BITNET_PLANES` multi-plane packing), i2s GGUF
  import, the vendored NeoGPU ternary f32 NEON kernel (MIT, verbatim) exposed through FFM, JNI and
  Kotlin/Native including a fused lm_head kernel, requant adapters, NEON BitNet packing, and ternary
  benchmarks + getting-started docs.
- **Iris dataset provider** ([#1044](https://github.com/SKaiNET-developers/SKaiNET/issues/1044),
  [#1101](https://github.com/SKaiNET-developers/SKaiNET/pull/1101), contributed by @AjithGoveas):
  the embedded 150-row dataset used by the new Android classifier tutorial.
- Sliding-window KV SDPA ([#1036](https://github.com/SKaiNET-developers/SKaiNET/issues/1036)) and
  KV formats declared by the store ([#1077](https://github.com/SKaiNET-developers/SKaiNET/issues/1077)).

### Fixed

- Packed block order end to end: feed-order bytes in a type claiming canonical order decoded to
  plausible garbage ([#1124](https://github.com/SKaiNET-developers/SKaiNET/issues/1124),
  [#1126](https://github.com/SKaiNET-developers/SKaiNET/pull/1126)); `TensorData` now declares its
  `BlockOrder` and every reader agrees on the same bytes.
- Packed ternary weights reach dispatch through the Wᵀ marker — ANY packed block storage routes through
  the marker path ([#1136](https://github.com/SKaiNET-developers/SKaiNET/issues/1136),
  [#1181](https://github.com/SKaiNET-developers/SKaiNET/pull/1181)).
- M2 acceptance page-fault flake ([#1107](https://github.com/SKaiNET-developers/SKaiNET/pull/1107));
  safetensors dtype mapper no longer prints a WARNING into stdout mid-parse (#1169).

### CI & process

- `apiCheck` has its own named PR leg (`test (api-compatibility)`) instead of hiding inside
  golden-parity ([#1176](https://github.com/SKaiNET-developers/SKaiNET/pull/1176)), after a stale dump
  reached `develop` unnoticed ([#1174](https://github.com/SKaiNET-developers/SKaiNET/pull/1174));
  branch protection on `develop` now requires the full `build-job` aggregator, admins included.
- Android per-target API dumps dropped (jvm + klib only,
  [#1111](https://github.com/SKaiNET-developers/SKaiNET/pull/1111)).

### Docs

- The memory model as built: `explanation/memory-model.adoc`, `explanation/packed-weight-layout.adoc`
  ([#1106](https://github.com/SKaiNET-developers/SKaiNET/pull/1106) — design drafts under
  `docs/design/` retired in favour of Antora pages), `explanation/virtual-tensors.adoc` (the ML
  Drift-style split, with diagrams), and SKEEP-003a (`skeep/003a-placement-and-planning-resolution.adoc`)
  recording the P7/P8 resolutions.
- Tutorials whose code cannot rot: the Android classifier getting-started and the ternary
  getting-started, with snippets compiled *and executed* in CI by `skainet-docs-samples` (the Iris
  training loop asserts held-out accuracy ≥ 0.80).
- SKEEP-003 accepted ([#932](https://github.com/SKaiNET-developers/SKaiNET/issues/932)) with its
  thirteen design decisions; how-to: plan a model's memory before loading it, including the
  embedded-device (`edge`) verdict.

## [0.40.1] - 2026-08-12

Headline: **correctness hotfix — silently wrong output, not a crash.** `DefaultCpuOps.transpose()` for packed quantized weights (Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K) performed a shape-only relabel instead of a real block-grid byte permutation whenever a row spanned more than one quant block (`blocksPerInputDim > 1` — true of virtually every real model). `ops.matmul(x, ops.transpose(W))` fed the packed-quant kernels bytes in the wrong order across all three kernel tiers — scalar, Panama-vector, and native (FFM/JNI) — silently producing wrong numbers, sometimes all-zero output, with no exception raised. Upgrading is strongly recommended for anyone using packed-quantized weights with `ops.transpose()`.

### Fixed

- **Packed-quant `transpose()` silently corrupted matmul output**
  ([#968](https://github.com/SKaiNET-developers/SKaiNET/issues/968),
  [#969](https://github.com/SKaiNET-developers/SKaiNET/pull/969))
  — `DefaultCpuOps.transpose()` swapped the tensor's shape metadata without
  physically reordering the underlying `packedData` bytes, on the assumption
  that the packed-quant matmul kernels index those bytes block-major
  regardless of layout. That assumption only holds when there is a single
  quant block per row (`blocksPerInputDim == 1`); for any wider row the
  canonical (row-major) and kernel-native block orderings are literal
  transposes of the `(outputDim, blocksPerInputDim)` block grid and do not
  coincide, so a freshly-loaded weight run through `ops.transpose()` fed the
  scalar, Panama-vector, and native (FFM/JNI) kernel tiers alike bytes in the
  wrong order — for all seven packed formats (Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K).
  `transpose()` now performs a real `O(bytes)` block-grid permutation
  (`transposePackedBlocks`); a misaligned packed tensor (`inputDim` not a
  multiple of the format's block size) now throws `IllegalArgumentException`
  instead of silently truncating a partial trailing block.
  `DefaultCpuOpsJvm`'s separate, independently-buggy shape-swap-only
  interception for `Q4_KTensorData` is removed, falling through to the
  shared corrected implementation. Caught by a new ground-truth regression
  test (`NativeLazyTransposeGroundTruthReproTest`) that dequants each packed
  format's canonical and kernel-native byte layouts independently and checks
  both the classic (`transpose` + matmul) and pre-transposed paths against
  that ground truth, per format — the kind of test the original "same bytes,
  new shape" optimization lacked.

## [0.40.0] - 2026-08-11

Headline: **big models fit on real devices, and SKaiNET reaches iOS/macOS
natively.** Off-heap/mmap tensor storage lets Android load models beyond the
hard ART heap cap by paging weight bytes from mapped files instead of the
managed heap, and a compounding GGUF dequantization bug that transiently
needed >12 GB heap for a 1.1B Q4_K_M model is fixed down to a ~1.05x-of-dense
floor. Q5_0/Q5_1 packed matmul reaches the native tier (FFM, Kotlin/Native,
JNI) for the first time, and `skainet-backend-native-cpu` now publishes
iOS/macOS Kotlin/Native targets whose single Apple arm64 archive dispatches
FEAT_DotProd at runtime — one build serves A12 through M-series.

### Added

- **Off-heap / mmap tensor storage on Android**
  ([#921](https://github.com/SKaiNET-developers/SKaiNET/issues/921), SKEEP-002/SKEEP-003
  slice): the java.nio memory-mapped storage that existed jvmMain-only is now shared
  source between the JVM and Android compilations (`FileChannel.map` is API 1 — no JNI):
  `MmapFloatTensorData`/`MmapTensorSource` (skainet-lang-core), `JvmMappedMemoryChunk`,
  `MappedRandomAccessSource` and the `BufferHandle.FileBacked` resolver
  `JvmFileBackedResolver` (skainet-io-core). New `MappedGgufWeights` (skainet-io-gguf,
  JVM+Android) opens a GGUF once, maps it read-only, and serves dense F32 tensors as
  zero-heap mapped views, any tensor as a `FileBacked` `TensorStorage` descriptor, and
  packed payloads as heap bytes for the existing kernels. Weight bytes live in file-backed
  pages the OS pages in/out — outside the hard ART heap cap that limited practical model
  size on Android. Verified host-side (no device required): a 640 MB dense model loads and
  reads through mapped views with **1.4 MB** of managed-heap allocation (0.0022x of the
  dense size; per-thread allocation counters), and the Android compilation is exercised by
  new `androidHostTest` suites (96 MB payload, ~240 KB used-heap growth). Files over 2 GB
  are rejected fast (single-region mapping); windowed mapping is a follow-up under
  SKEEP-003's IO pipeline improvement.
- **Native Q5_0 / Q5_1 packed matmul kernels (FFM, Kotlin/Native, JNI).** 0.39.0 shipped
  packed GGUF *loading* for Q5_0/Q5_1 plus scalar + Panama kernels, but the native tier had
  no Q5_x kernels — on the JVM the registry cascaded to Panama (50), and on Kotlin/Native and
  Android the formats ran on the priority-0 scalar floor. New `skainet_q5_0_matmul` /
  `skainet_q5_1_matmul` C kernels (plain NEON, no dotprod/i8mm requirement — runs on every
  AArch64 core) expand the `qh` high-bit plane with a per-lane `vtstq_u8` bit test and fold
  the dequant algebraically (`d*(dot - 16*Σx)` for Q5_0, `d*dot + m*Σx` for Q5_1) so the
  per-block input sum hoists out of the output-row loop. Wired into all three consumers:
  the FFM `NativeKernelProvider` (JVM), the cinterop `NativeKnKernelProvider`
  (Kotlin/Native), and the Android JNI bridge (`JniKernels.q50Matmul`/`q51Matmul` +
  `JniKernelProvider`), each with parity tests against the scalar references. Unblocks the
  packed Q5_1 path for `functiongemma-270m` "Q5_K_M" checkpoints (whose attention/FFN
  weights are Q5_1) under `NATIVE_OPTIMIZED` — see SKaiNET-transformers#170. (#708)
- **`skainet-backend-native-cpu` publishes Apple Kotlin/Native targets** — `iosArm64`,
  `iosSimulatorArm64`, and `macosArm64` klibs with the Mach-O kernel static archive embedded
  via cinterop, exactly like the Linux pair ([#959](https://github.com/SKaiNET-developers/SKaiNET/issues/959),
  iOS kernel track of [#920](https://github.com/SKaiNET-developers/SKaiNET/issues/920)).
  Archives are built by new Apple CMake lanes on a macOS host (platform SDKs, no `-march` —
  the #958 runtime FEAT_DotProd dispatch serves A12 through M-series from one device archive)
  or injected in CI via `-PskainetKernelsIosArm64Dir` / `-PskainetKernelsIosSimulatorArm64Dir` /
  `-PskainetKernelsMacosArm64Dir`. The shared nativeTest parity suites now also run as
  `macosArm64Test` (native) and `iosSimulatorArm64Test` (simulator) on the macos-14 PR lane.
  On non-macOS hosts the Apple task family is disabled (ubuntu CI unaffected). Registration
  on K/N remains manual via `installNativeKernels()` — Apple consumers call it once at startup,
  same as Linux.

### Fixed

- **GGUF `DEQUANTIZE_TO_FP32` no longer over-allocates**
  ([#782](https://github.com/SKaiNET-developers/SKaiNET/issues/782)): loading a 1.1B Q4_K_M
  transiently needed >12 GB heap against a ~4.4 GB dense-FP32 floor. Three compounding causes,
  all in `skainet-io-gguf`: (1) the legacy `GGUFReader` eagerly materialized **every** tensor
  payload as a boxed `List<Any>` at parse time — measured at **41x** the payload size in
  allocations (~26 GB for a 637 MB file); payloads are now constant-space lazy views that
  decode elements on access (same `List<Any>` API, same contents). (2) every dense tensor paid
  a full-size defensive `copyOf` in the tensor factory on top of the dequant intermediate —
  `StreamingGgufParametersLoader` now wraps its loader-owned arrays zero-copy
  (`ctx.wrapFloatArray`). (3) the K-quant kernels allocated per-block `copyOfRange` scratch —
  they now index the source buffer directly, so a full-tensor dequant allocates exactly the
  destination `FloatArray`. `StreamingGgufParametersLoader` also gains an optional
  `quantPolicy` parameter: `DEQUANTIZE_TO_FP32` streams each quantized tensor block-by-block
  straight into its destination array (peak transient per tensor = the packed source bytes;
  measured: eager FP32 load of a synthetic multi-tensor model allocates 1.38x the FP32 total
  vs 2.1-2.3x for the historical copy chain, with peak live ≈ 1.05x). The default
  (`NATIVE_OPTIMIZED`) keeps the loader's historical packed-block behavior bit-for-bit; a
  parity test pins the dequant path to the packed accessors bit-exactly across all seven
  supported quant formats.

### Performance

- **Apple arm64 runtime FEAT_DotProd dispatch for the Q4_K/Q6_K C kernels**
  (`skainet-backend-native-cpu`, [#958](https://github.com/SKaiNET-developers/SKaiNET/issues/958),
  part of the iOS kernel track of [#920](https://github.com/SKaiNET-developers/SKaiNET/issues/920)).
  Apple builds now compile at the SDK-default arm64 baseline — a Kotlin/Native klib embeds
  exactly one static archive, and Apple A12 (iPhone XS/XR, still iOS-supported) lacks
  FEAT_DotProd while A13+/M-series have it — with the dotprod hot bodies compiled twice
  (baseline + `target("dotprod")`-attributed) and selected once per matmul via a cached
  `sysctlbyname("hw.optional.arm.FEAT_DotProd")` probe. Non-Apple builds keep the compile-time
  `-march` guard as the only mechanism; Linux codegen is unchanged (qemu parity green, `sdot`
  verified in the cross archive). The existing macOS FFM dylib moves from TU-level dotprod to
  baseline+dispatch — runtime-equivalent on every Apple Silicon Mac. iOS builds are static-only
  (`SKAINET_STATIC_ONLY`, auto-on for `CMAKE_SYSTEM_NAME=iOS`).

### CI

- **Releases embed the Apple kernel archives** ([#959](https://github.com/SKaiNET-developers/SKaiNET/issues/959)):
  publish.yml's macos leg builds the three Mach-O static archives (with an `nm`/`objdump`
  `sdot` assertion guarding the #958 dispatch body against clang's silent unknown-feature
  ignore), uploads them fail-loud, and the publish job verifies and injects them via the
  `-PskainetKernels{IosArm64,IosSimulatorArm64,MacosArm64}Dir` properties — the same
  verified-artifact-or-fail contract as the Linux ELF archives.

### Documentation

- **Kernel support matrix gains the `native-cinterop` tier** (Native·linux + Native·apple,
  the 7 packed-quant formats) — the Native·linux column was under-reported as `scalar`
  before; both native columns now reflect `NativeKnKernelProvider`
  ([#959](https://github.com/SKaiNET-developers/SKaiNET/issues/959)). The eager-backends
  mindmap and the `installNativeKernels()` KDoc document the manual-registration contract
  and the Apple A12 dispatch fallback.

## [0.39.1] - 2026-08-11

Headline: **eager overhead off the JVM is gone.** The eager CPU ops gain
primitive FP32 fast paths, removing the per-element allocation/boxing overhead
that dominated on-device LLM decode (83% of end-to-end time on a Pixel 8a even
with NEON matmul), and `DirectCpuExecutionContext.ops` is cached instead of
rebuilt per access. The README now points LLM users to SKaiNET-transformers.

### Documentation

- **README points LLM users to SKaiNET-transformers**
  ([#923](https://github.com/SKaiNET-developers/SKaiNET/issues/923)): a callout under
  "Start in 5 minutes" says plainly that LLM inference lives in the
  SKaiNET-transformers repository — this repo is the engine underneath — and names
  the `sk.ainet.transformers` artifacts and BOM to depend on.

### Performance

- **Primitive FP32 fast paths for the eager CPU ops** (`skainet-backend-cpu`,
  [#949](https://github.com/SKaiNET-developers/SKaiNET/issues/949)). The generic paths in
  `DefaultCpuOps` paid, per element: two `IntArray` allocations for broadcast index mapping, a
  vararg-spread boxed `data.get`, a KClass `when (dtype)` comparison, and a boxed lambda
  round-trip. On ART that overhead dominated LLM decode — 83% of end-to-end SmolLM2-135M decode
  on a Pixel 8a was non-matmul overhead even with the NEON backend doing every matmul. The hot
  ops now run flat primitive loops over the dense `FloatArray` buffer (falling back to the
  generic path for other dtypes/layouts): binary/scalar arithmetic (incl. last-dim bias
  broadcast, mirroring the JVM vector path's coverage), the activation family
  (relu/sigmoid/silu/gelu/…), unary math (sqrt/exp/log/sin/cos/tanh/pow/…), softmax and
  logSoftmax along the last dim (also removing an O(n²)-per-slice max/denominator recompute),
  sum/mean reductions, concat (block copy), and reshape/flatten (buffer copy). Benefits every
  non-JVM target — Android, Kotlin/Native, JS/Wasm; the JVM ops class keeps its Panama/FFM
  specializations on top.
- **`DirectCpuExecutionContext.ops` is cached.** The getter previously constructed a fresh ops
  instance on every access, re-running per-instance lazy kernel resolution in the eager hot
  loop ([#949](https://github.com/SKaiNET-developers/SKaiNET/issues/949)).

## [0.39.0] - 2026-08-10

Headline: **on-device AI on Android becomes real.** A JNI NEON kernel backend
(`skainet-backend-jni-cpu`) brings hand-tuned ARM matmul to Android — where the
FFM provider can never run — measured at ~24 tok/s SmolLM2-135M Q8_0 decode on a
Pixel 8a versus ~3.8 scalar (6.4x), clearing the on-device usability bar. The
release also hardens the GGUF load path on Android (streaming instead of
full-file heap loads), makes published Kotlin/Native kernel klibs linkable, and
lands a batch of tensor-storage correctness fixes.

### CI

- **Release workflow publishes the `skainet-backend-jni-cpu` AAR.** `./gradlew publish` now
  includes the Android JNI kernel module, whose AAR carries NDK-cross-built `.so`s; the publish
  job gains Android SDK + a pinned NDK setup so the native build succeeds on the release runner
  (previously the release would ship no JNI artifact, or fail). The NDK version is pinned in
  `gradle/libs.versions.toml` (`android-ndk`) and referenced by the module's `ndkVersion` and the
  workflow, so local and CI builds are reproducible. (#946)

### Added

- **NEON body for the Q4_0 matmul kernel.** `skainet_q4_0_matmul` was the only priority
  quant format without a SIMD path (scalar C only, while q8_0/q4k/q5k/q6k had NEON).
  It now unpacks the split-layout nibbles with `vand`/`vshr`, re-centres in the signed
  int8 domain, and widens to f32 FMA lanes — plain NEON with no dotprod/i8mm requirement,
  so it runs on every AArch64 core, and the same block-outer/row-inner loop order as
  q8_0 (sequential weight reads; per-row accumulation order unchanged). Verified: 27/27
  kernel tests green under qemu-aarch64 (cross-built `-march=armv8.2-a+fp16+dotprod`,
  K/N-bundled gcc 8.3), `fmla` confirmed in the archive's disassembly; a new
  Kotlin/Native Q4_0 parity test closes the gap where the aarch64 lane had no Q4_0
  coverage at all. Part of the mobile-kernels effort (#920).
- **`skainet-backend-jni-cpu`: Android JNI bridge for the native NEON kernels.** ART has no
  `java.lang.foreign`, so the priority-100 FFM provider can never run on Android — until now
  Android inference ran on the priority-0 scalar floor. The new AAR ships the same C kernel
  sources via NDK/CMake with thin JNI shims (`GetPrimitiveArrayCritical`, zero-copy pins) and
  a priority-100 `JniKernelProvider` discovered via `ServiceLoader` (which ART supports; the
  Android ops factory now installs discovered providers exactly like the JVM does). Two `.so`
  tiers are built from the same sources and selected at load time from `/proc/cpuinfo`:
  baseline `armv8-a` (NEON, runs on every arm64 core — 0 dot-product instructions, verified by
  disassembly) and `armv8.2-a+fp16+dotprod` (enables the `vdotq_s32` q4k/q6k paths — would
  SIGILL on Cortex-A53-class cores, hence the gate). Q8_0/Q4_0/Q4_K/Q5_K/Q6_K bridged;
  16 KB-page-aligned `.so`s (Android 15+); consumer R8 rules keep the ServiceLoader entry in
  release builds. On-device parity tests included (`androidTest`, vs the scalar references).
  Part of the mobile-kernels effort (#920).

### Fixed

- **GGUF tensors report their real encodings and sizes in `TensorStorage`.**
  `StreamingGGUFReader` mapped only Q4_K/Q8_0 to dedicated `TensorEncoding`s; Q4_0, Q5_0,
  Q5_1, Q5_K and Q6_K — all of which have encoding objects and CPU kernels — fell through to
  `Opaque(name, 0)`, whose zero byte count short-circuits `TensorStorage.physicalBytes` and
  silently corrupted every memory report and compression ratio. All seven quant formats now
  map to their encodings, and genuinely unknown types carry the tensor's real byte count in
  `Opaque`. (#928)
- **Memory-copy diagnostics attribute copies to their source.** `MemoryTracker.recordCopy`
  discarded the `sourceName` every instrumented call site passes; reports now carry a
  per-source breakdown (`copiesBySource: Map<String, CopySourceStat>`, included in the
  report's text form, sorted by volume). `ActiveMemoryTracker.current` is now `@Volatile`
  with an honest thread-safety contract in its docs; making the tracker per-execution-context
  instead of a process-wide hook is part of the SKEEP-003 storage-model discussion. (#931)
- **`TensorStorageFactory` ownership labels are now truthful.** `borrowFloatArray` re-encoded
  the floats into a private byte copy and labeled it `Borrowed` despite its "(zero-copy)" doc —
  it is now deprecated (delegating to `fromFloatArray`) and honestly returns `Owned`;
  `fromTensorData`'s doc claimed "borrowed (not copied)" while its dense branches copy — the
  contract is now documented per branch (packed Q4_K/Q8_0 genuinely borrow zero-copy, dense
  arrays convert to owned bytes) and pinned by ownership + mutation-visibility tests. (#927)
- **`TensorStorage` transfer API can materialize its own placements.** `copyMaterialize()`
  threw for `Aliased` handles (now resolved directly, producing an independent owned copy of
  the slice) and for `FileBacked` — meaning `copyToHost()` could not bring `MMAP_WEIGHTS`
  storage to the heap, the one transfer the storage layer was designed around. Both methods
  gain a `BufferResolver` overload that reads file-backed regions through a configured
  resolver; `DeviceResident` remains unsupported with an error that says why. (#929)
- **Published Kotlin/Native klibs for `skainet-backend-native-cpu` now carry their machine
  code.** The static kernel archive was attached via project-local `linkerOpts`, which does
  not travel with a published klib — downstream K/N consumers of the `-linuxx64`/`-linuxarm64`
  artifacts could not link (unresolved `skainet_*` symbols). The archive is now EMBEDDED into
  the cinterop klib (`-staticLibrary`/`-libraryPath`), conditional on a correct-architecture
  ELF archive being available; bindings-only builds (e.g. macOS dev hosts) warn loudly instead
  of staying silent. The release workflow's ubuntu leg now also cross-builds and uploads both
  linux static archives, and the macOS publish job injects them, so released klibs embed real
  code. In-repo K/N test binaries link purely from the embedded klib — the consumer-link
  scenario is what the test suite now exercises. (#941)
- **`TensorData.copyToFloatArray()` default implementation works for rank >= 2.** It used to
  iterate a single flat index into the vararg `get`, tripping every implementation's
  one-index-per-dimension arity check — a latent trap for any implementation that didn't
  override it. The default now unravels flat positions into per-dimension indices (row-major);
  a contract test exercises the default at ranks 1–3. (#930)
- **Streaming GGUF loads fail fast on unsupported tensor types instead of silently skipping them.**
  `StreamingGgufParametersLoader` used to emit a `SKIP` progress string for any tensor type outside
  its `when` and deliver a model with silently missing weights — the failure then surfaced far away
  in the forward pass (the load-time half of the Q4_1 report in
  [#654](https://github.com/SKaiNET-developers/SKaiNET/issues/654)). An eager pre-scan of the tensor
  directory now throws `IllegalArgumentException` before any tensor is delivered, naming every
  offending tensor, its type (including raw values for unknown types), and the supported set; the
  per-tensor `else` is a hard error guarding against drift from the new
  `SUPPORTED_TENSOR_TYPES` companion set. **Behavior change:** files that previously "loaded" with
  skipped tensors now fail at load — the legacy `GgufParametersLoader` already behaved this way.
  Q4_0 / Q5_0 / Q5_1 — which had packed `TensorData` and matmul kernels but were missing from the
  loader — now load as packed blocks instead of being skipped. Closes
  [#919](https://github.com/SKaiNET-developers/SKaiNET/issues/919).
- **Random file access on Android: streaming model loads instead of full-file heap loads.**
  `createRandomAccessSource` unconditionally returned `null` on Android in `skainet-io-gguf`,
  `skainet-io-safetensors` and `skainet-io-onnx`, forcing every load through the legacy
  materialise-the-whole-file path — on a real device a 138 MiB GGUF then OOMs the ART heap
  (capped at 256/512 MB) before tensors are even built. A new `AndroidRandomAccessSource` in
  `skainet-io-core` (androidMain, positional `FileChannel` reads — thread-safe, API 1+) now backs
  all three actuals, making `StreamingGGUFReader`/streaming loaders reachable on Android; this
  also un-breaks `TokenizerFactory.fromGguf`, which needs `StreamingGGUFReader.fields`. Android
  host-side unit tests (`withHostTest {}`, a first in the repo) cover the read contract including
  concurrent positional reads. Closes [#922](https://github.com/SKaiNET-developers/SKaiNET/issues/922).

## [0.38.0] - 2026-07-30

### Added

- **Narrow-float (BF16 + FP16) weights kept packed.** A shared `NarrowFloatCodec` layer
  (`Bf16Codec`/`Fp16Codec`) plus `NarrowFloatDenseTensorData`/`Fp16DenseTensorData` let SafeTensors F16
  and GGUF F16/BF16 weights load `KEEP_NATIVE` — two bytes per element at rest instead of widening to
  FP32 at load. `DefaultCpuOpsJvm` dispatches to format-specific matmul kernels by codec, so the packed
  weight reaches the kernel rather than a widened copy. Narrow floats are a *storage* width only:
  kernels widen to f32 lanes, accumulate in f32, and narrow on store.
- **Native (FFM) FP16 matmul kernel.** `skainet_fp16_matmul` joins the existing BF16 kernel in
  `skainet-backend-native-cpu`, wired through `NativeKernelProvider.matmulFp16()`. Until now the
  provider carried `matmulBf16` but no FP16 counterpart, so BF16 resolved to the native kernel at
  priority 100 while FP16 silently cascaded to the JVM Panama kernel at 50 — which read as a slow
  kernel and was a missing one. `KernelProvider.supports` gains the matching `"Float16"` arm, absent
  while every other matmul dtype was present.

- **First-class dynamic dimensions (`Dim`).** A new `sk.ainet.lang.tensor.Dim` vocabulary makes "dynamic
  extent" explicit instead of an overloaded `-1`: `Dim.DYNAMIC` is a reserved sentinel (`Int.MIN_VALUE`)
  **distinct from reshape's `-1` = infer**, with the dynamic-aware shape arithmetic (`concat`, `compatible`,
  `render`, `isDynamic`/`isStatic`) centralized in one place rather than scattered `extent < 0` guards.
  `Shape` gains `hasDynamic()`, `isDynamic(axis)`, `dynamicAxes`, and its `volume` now throws on a dynamic
  shape (an unknown extent has no materializable element count) instead of returning a corrupt product.
  The slice/range DSL is dynamic-aware too: `all()` over a dynamic axis is a valid symbolic full-axis, and
  reshape passes a dynamic target through unchanged. The `skainet-compile-hlo` emitter now shares this one
  sentinel (`TypeMapper.DYNAMIC_DIM` aliases `Dim.DYNAMIC`), so tracer and emitter agree by construction.
- **Dynamic-shape-safe StableHLO emission (streaming KV-cache decode).** The `skainet-compile-hlo`
  emitter now renders a `-1` tensor extent as an MLIR `?` (dynamic) dimension and emits op forms that
  `iree-compile` accepts under a dynamic dim, so a single compiled vmfb serves every autoregressive
  decode step (growing KV cache) instead of one fixed cache length. A new `TypeMapper.DYNAMIC_DIM = -1`
  marker plus a `List<Int>.hasDynamic()` predicate gate the dynamic paths, so every static graph is
  emitted byte-for-byte unchanged. Verified: one dynamic SDPA vmfb runs at key lengths 3 and 17, and
  the full FunctionGemma `with_past` decode graph (real weights, dynamic `1x{nKV}x?x256` cache)
  self-compiles from the DSL to a CPU vmfb — a graph that could not be compiled before.
- **Dynamic-shape-safe trace finalization.** `TraceToGraphBuilder.extractFloatArray` no longer probes
  `Tensor.volume` for non-dense data, so a dynamic-shaped graph input (e.g. a `?` KV-cache tensor) is left
  as an input instead of tripping the (correctly) throwing `volume` on an unknown extent. This lets a decode
  graph with dynamic caches finalize to a `ComputeGraph` and compile — verified with the real Moonshine v2
  `with_past` decoder (dynamic self *and* cross caches, `1x8x?x40`) self-compiling from the DSL to a vmfb.
- **Allocation-free shape-only tracing (`VoidTensorOps`).** The trace-time op set now propagates shapes
  through a `ShapeOnlyTensorData` that carries a `Shape` but allocates no backing buffer, so a dynamic
  (`-1`) extent flows through a whole decode trace instead of throwing `NegativeArraySizeException` when
  a real buffer of negative size is allocated. This is what lets a real KV-cache seq dim be traced as
  dynamic end-to-end (rather than via a sentinel-dimension + post-emit text substitution).

### Changed

- **SDPA scale folded into Q.** `AttentionOperationsConverter` now applies the attention scale as a
  scalar constant multiplied into Q *before* the QK `dot_general` (`(q·s)@kᵀ ≡ scores·s`, exact),
  instead of a dense splat constant sized to the full scores shape. This drops a scores-sized constant
  from every attention graph and, crucially, avoids an invalid dynamic-shape splat when the key/cache
  dim is `?`.
- **Softmax broadcasts are dynamic-shape-safe.** When the softmax/scores shape carries a dynamic dim,
  `AttentionOperationsConverter` and `ActivationOperationsConverter` broadcast the reduced max/sum with
  `stablehlo.dynamic_broadcast_in_dim` (runtime shape operand built via `get_dimension_size` +
  `concatenate`) instead of a static `broadcast_in_dim`. Static graphs keep the explicit
  `broadcast_in_dim` unchanged.
- **Identity reshapes/slices are elided.** `ShapeOperationsConverter` returns the operand SSA value with
  no emitted op when a reshape's input and result types are identical, or a `slice`/`narrow` covers the
  full extent of every axis (the KV-cache "cache-as-output-sink" and full-cache head-expansion patterns).
  Besides being a no-op, this is the only valid lowering on a dynamic axis — a static `stablehlo.slice`
  cannot express a full-extent bound on a `?` dim (its limit would be the `-1` extent, e.g. `0:-1:1`).
- **Concatenate propagates dynamic extents.** Both the trace-time shape inference
  (`VoidTensorOps.calculateConcatShape`) and the emitter (`ShapeOperationsConverter` concat) now keep the
  concatenated axis dynamic when any operand's extent there is dynamic, instead of numerically summing it
  (which turned a growing cache `? ++ 1` into a bogus static `0`).
- **Narrow-float weights transpose for free.** `NarrowFloatInputMajorTensorData` stores a rank-2 narrow
  weight input-major, so the `[out, in]` → `[in, out]` transpose that `Linear.onForward` performs on
  every call is a zero-copy reinterpretation of the same buffer. Projections are stored `[out, in]` but
  the narrow matmul dispatch needs `[in, out]`, and transposing a row-major narrow tensor previously
  walked it elementwise through boxed `get()` and widened to FP32 — 4.4 s for a 4096x11008 projection,
  per weight, per token, which made `KEEP_NATIVE` slower than not using it. A row-major narrow buffer
  deliberately still takes the generic path: swapping its shape would silently yield a different matrix
  rather than the transpose.
- **Native narrow-float kernels read the weight once per matmul.** Both the BF16 and FP16 kernels tile
  the `j` dimension at `m > 1` and widen each weight row once per tile, instead of walking the whole
  weight matrix once per row of the input. Accumulation into any output element stays `p` ascending, so
  results are bit-identical to the previous formulation. At `m == 1` both keep the straight pass — there
  is nothing to amortize and tiling costs ~15% there.
- **`Fp16Codec.decode` is straight-line.** The subnormal arm no longer renormalizes in a data-dependent
  loop; binary16 subnormals are `mant * 2⁻²⁴` with both factors exact in FP32, so one multiply lets the
  hardware renormalize. A NaN now decodes **quiet**, matching `encode` (which already never emitted a
  signaling binary16 NaN) and the hardware conversion the JVM kernel uses. This changes 1022 patterns —
  the signaling NaNs — and nothing else; without it the JVM and every other target would disagree on
  them.

### Fixed

- **FP16 matmul was 2–18x slower than the FP32 SGEMM it replaces.** The cause was dispatch, not
  arithmetic: `NativeKernelProvider` had no `matmulFp16()`, so FP16 fell through to the JVM kernel while
  BF16 ran natively. Head to head the two JVM kernels are within ~15% of each other. FP16 is now
  1.5–1.7x *faster* than FP32.
- **CI and supply-chain hardening.** Least-privilege permissions on the build workflow, the docs MathJax
  npm install pinned by version, and a `logback-classic` bump.

### Performance

Measured on an i7-9750H (AVX2), OpenJDK 21, median ms per call, `4096x11008` projection:

| | batch 1 | batch 16 |
|---|---|---|
| FP32 SGEMM | 108.7 | 271.1 |
| BF16 | 57.4 | 143.5 |
| FP16 | 71.2 | 159.3 |

Both narrow formats now beat the FP32 SGEMM — BF16 by 1.8–1.9x, FP16 by 1.5–1.7x. Note these kernels
are compute-bound on the FMA chain at batch 16, not bandwidth-bound: cutting weight traffic 16x bought
only 9–19%, so the next win is a blocked microkernel or `bfdot`/`bfmmla` on ARMv8.6-A+, not more layout
work.

## [0.37.0] - 2026-07-25

### Added

- **`Lstm` layer.** Single-layer, batch-first LSTM mirroring `Gru`'s unroll-at-trace-time design and
  built from existing primitives only (matmul/narrow/sigmoid/tanh/multiply) — no new `TensorOps` op,
  and it traces to StableHLO with no dedicated converter. Gate order `i,f,g,o` and dual biases match
  `torch.nn.LSTM`. Adds `LstmState(h, c)` with an explicit caller-owned `step(xt, state, ctx)` API —
  required by transducer prediction networks (RNN-T/TDT) and exactly the shape that lowers to a
  fixed-shape single-step StableHLO graph — plus an `initialState(batch, ctx, dtype)` helper. (PR #824)
- **Learning-rate schedules and mutable optimizer `lr`.** `AdamOptimizer` and `SgdOptimizer` expose
  `lr` as a public `var`, so training loops can adjust it between steps without recreating the
  optimizer and losing the moment estimates. A stateless `LrSchedule` fun interface plus a
  `linearWarmupCosineDecay` factory cover the GPT-style pretraining recipe: linear ramp from
  `initialLr` to `peakLr` over the warmup steps, then cosine decay to `minLr`. (PR #866, issue #865)
- **BREAKING (Java callers): optional bias in `Linear`.** `initBias` is now nullable with a `null`
  default, creating a bias-less projection (`y = x W^T`) — the equivalent of PyTorch's
  `nn.Linear(bias=False)`, needed for GPT-style `qkv_bias=false` projections and weight-tied output
  heads. A bias-less layer registers only its weight parameter, so parameter counts and checkpoints
  match architectures defined without bias. Adds a `biasOrNull()` helper next to the throwing
  `bias()` accessor. **The `@JvmOverloads` overload set changes:** the 4-arg
  `(in, out, weights, bias)` convenience overload is replaced by `(in, out, weights)`, so Java code
  binding the 4-arg overload fails to compile (and breaks binary compatibility) until it moves to the
  full constructor. Kotlin callers bind the full constructor and are unaffected. (PR #870)
- **`Linear` is `open`.** Adapter-style layers (LoRA and similar) can subclass it and augment
  `onForward` or `params` while reusing the base projection. No behavior change — the override points
  were already open via `Module`. (PR #875)
- **`androidNative` targets for the IO modules.** `skainet-io-core` gains `androidNativeArm64` and
  `androidNativeArm32` (via a 64-bit-only `native64Main` source set), and `skainet-io-safetensors`
  gains the `androidNative` target set; the latter also drops unused `compile-core`/`compile-dag`
  dependencies. (PRs #836, #842, #845)
- **OpenSSF Scorecard workflow and badge.** (PR #814)

### Changed

- **`Dropout` now actually drops.** It was an identity placeholder in both phases; it now applies
  inverted dropout under a training-phase context — each element is zeroed with probability `p` and
  survivors are scaled by `1/(1-p)`, so the expected activation is preserved and inference needs no
  rescaling. The mask is a constant tensor combined with an element-wise multiply, so gradients flow
  to surviving inputs without a dedicated autograd rule; an injectable `Random` enables reproducible
  masks. Models that relied on `Dropout` being a no-op will see different (correct) training-time
  activations. (PR #867, issue #861)
- **Toolchain and dependency bumps.** Kotlin 2.4.10 (JVM plugin and `kotlinx.serialization` plugin
  aligned), AGP 9.3.1, Shadow 9.6.1, Kover 0.9.9, JUnit Jupiter 6.1.2, and Logback 1.6.0. No public
  API or behavior changes. (PRs #818, #819, #820, #826, #829, #840, #856, #873, #874, #811)
- **Supply-chain hardening of CI and the docs image.** All GitHub Actions across every workflow are
  now pinned to commit hashes, and the documentation `Dockerfile` pins its base image by digest and
  its npm packages (Antora CLI, site-generator, lunr-extension, mermaid-cli, asciidoctor-kroki) to
  exact versions from the npm registry, making documentation builds reproducible. (PRs #816, #821,
  #827, #830, #831, #832, #834, #837, #838, #846, #848, #868)
- **CI `allTests` split into parallel per-target jobs**, ending the recurring out-of-memory flakes on
  the combined job. (PR #854)
- **Docs deployment publishes on release**, and fork PRs no longer fail on the comment step. (PR #850)

### Fixed

- **`scaledDotProductAttention` silently discarded the attention pattern at its default scale.** The
  op's `scale` parameter defaults to `0f`, documented as meaning `1/sqrt(headDim)`, but the CPU
  backend applied it literally — every score was multiplied by zero, so the softmax collapsed to a
  uniform average. The CPU kernel now resolves `scale == 0f` to `1/sqrt(headDim)`. Any model calling
  SDPA without an explicit scale was producing attention-free output. (PR #880, issue #860)
- **Three autograd bugs that silently froze or crashed training.** (PR #877, issues #862, #863, #864)
  - `CrossEntropyLoss`: the index-target path built its result host-side via
    `tensorDataFactory` + `fromData`, detaching the tape so no gradient reached the predictions, and
    the soft-target path only recorded when targets lived on the recording context. Both now compute
    the NLL with differentiable ops dispatched through the predictions' ops (a constant one-hot for
    the index path), keeping the tape connected.
  - `softmax`/`logSoftmax` backward: a negative `dim` (e.g. `softmax(dim = -1)`) was passed straight
    to `broadcastToInput`, which reserves `-1` as a no-unsqueeze sentinel, so the reduced axis was
    never re-expanded and backward crashed for rank ≥ 3. The dim is now normalized.
  - `varianceBackward`: the reduced mean and upstream kept the reduced shape, so the
    subtract/multiply could not broadcast for rank ≥ 2, and `N` used the total volume instead of the
    axis size. Both are now expanded over the reduced axis and `N` is the axis size.
- **`argMax` traced through the `dag {}` builder emitted invalid StableHLO.**
  `GraphDsl.inferDagOutputSpecs` had no `argMax` case, so it echoed operand-0's shape and dtype; the
  converter then faithfully emitted an integer literal for an `f32` tensor (rejected by
  `iree-compile`) and a reduce whose result kept the reduced dim. The builder now infers a reduced
  shape with an `Int32` index dtype, matching the `VoidTensorOps` path that was already correct —
  which is why only the raw `dag {}` path broke. (PR #878, issue #876)
- **Legacy `tokenizer.json` files without `model.type` failed to load.** Files such as
  `openai-community/gpt2` omit the field; `TokenizerFactory.fromTokenizerJson` now infers the type
  from structure — a `merges` list is unique to BPE — and routes them to
  `QwenByteLevelBpeTokenizer` instead of throwing. (PR #879, issue #858)
- **CPU `gather` threw on multi-dimensional indices.** An `[N, L]` index tensor needs one coordinate
  per dimension for a flat `data[i]` access; indices are now read in row-major order via the
  contiguous buffer, falling back to unravel. (PR #879, issue #859)

## [0.36.0] - 2026-07-11

### Added

- **REUSE / SPDX license-compliance setup.** Added `REUSE.toml` and the `LICENSES/` directory, a
  `reuse-compliance` CI workflow (running on `main` and `develop`), and a REUSE status badge in the
  README, making the repository machine-verifiable against the [REUSE](https://reuse.software/)
  specification. (PRs #806, #807, #808)

### Changed

- **Kotlin 2.4.0 toolchain.** The build now targets Kotlin 2.4.0 (up from 2.3.21), with the
  supporting toolchain aligned to match: KSP 2.3.10 and Dokka 2.2.0 (Dokka 2.1.0 is incompatible
  with the Kotlin 2.4.0 Gradle plugin). No public API or behavior changes — this is a compiler and
  build-toolchain upgrade. (PRs #658, #659, #660)
- **`skainet-data` module POM coordinates and names aligned with module names.** Data-module POM
  `artifactId`s and display names now match their Gradle module names, removing the previous
  mismatches. (PR #793)

### Fixed

- **`ComputeGraphExecutor` replayed `permute` as a plain transpose, dropping the recorded axes.**
  The builtin dispatch grouped `permute` with `transpose`/`transpose2d`, so a traced
  `permute(t, axes)` executed as a last-two-dims swap — only coincidentally correct for rank-2.
  Any rank-3 permutation (e.g. multi-head attention's heads/sequence swap `[1, 0, 2]` in a
  full-sequence encoder or prefill trace) silently produced the wrong layout; single-token decode
  paths never hit it, which is why generation workloads didn't surface the bug. `permute` now
  replays with its recorded `axes` (the same `List<Int>` convention `permuteBackward` and the
  StableHLO converter already parse), falling back to the legacy transpose behavior for older
  traces without axes. Surfaced by the BERT DSL-path encoder work in SKaiNET-transformers, which
  carries a temporary `LLMFusedOpHandlers` override to be removed once this fix ships.

## [0.35.0] - 2026-07-08

### Added

- **`argMax(tensor, dim)` tensor op.** Index of the maximum value along a dimension, with ties
  resolved to the **lowest** index (numpy/greedy); the reduced dimension is removed (no keepdim).
  Non-differentiable by design (index selection). Like `scaledDotProductAttention`, it stays a
  single op and is lowered at the **StableHLO stage** rather than needing a new primitive:
  `ArgMaxOperationsConverter` composes `iota` + reduce-`maximum` + `broadcast_in_dim` + `compare EQ`
  + `select` + reduce-`minimum` (the same `stablehlo` primitives the causal-mask code already
  emits) — so no variadic reducer region and no new DSL ops. The eager CPU kernel is a scalar
  reduction-to-index. Indices are `i32` in the compiled StableHLO; the **eager** path materializes
  them as index-valued floats in the input dtype so the result is a portable `Tensor<T, V>` (an i32
  payload inside a float tensor is unreadable on Kotlin/Native + Wasm). Unblocks folding an LLM's
  `logits → token-ids` argmax tail into the DSL trace instead of a post-hoc MLIR rewrite. (PR #800)

- **BREAKING (coordinates): `skainet-data-simple` now publishes under its module name.** The artifactId
  changes from the mismatched `sk.ainet.core:skainet-data-basic` to `sk.ainet.core:skainet-data-simple`;
  update your dependency declarations when upgrading past 0.34.0 (BOM consumers only need the new
  artifactId). `skainet-data-transform` also gets a distinct POM name ("skainet data transforms" — it
  previously duplicated the datasets module's name); its coordinates are unchanged.

## [0.34.0] - 2026-07-05

### Added

- **New module `sk.ainet.core:skainet-data-source` — URI-backed data sources.** One declarative way to
  get raw data into SKaiNET from `file://`, `https://`, and Hugging Face (`hf://owner/repo/path` and
  `hf+https://…`) URIs: `DataSource` contracts + `DataSourceUriParser`, a `DefaultDataSourceResolver`
  (JVM: `JvmDataSourceResolver` with artifact materialization/caching via `CachePolicy`), streaming of
  source artifacts with kotlinx-io, and parameterizable Hugging Face auth (token provider — no
  hard-coded credentials). (PRs #784, #785)
- **Raw dataset parsers + suspendable data pipeline DSL.** `DataFormatParser` implementations for CSV,
  TSV, JSON arrays/objects, and JSON Lines (`.jsonl` / `.ndjson`) produce schema-carrying `RawDataset`s;
  `rawDataset { from(…); format(…); cachePolicy(…) }` builds a dataset straight from a source URI, and
  `dataPipeline<T>()` chains named, schema-aware `dataTransformer` stages as a suspend pipeline.
  See the new [data sources getting-started tutorial](docs/modules/ROOT/pages/tutorials/data-sources-getting-started.adoc). (PR #785)
- **Dataset operation views and richer batches (`skainet-data-api`).** `Dataset` gains deterministic
  seeded `shuffle(seed)`, `split(ratio, seed, stratified)` (label-stratified splitting), `filter` /
  index-based views, optional `inputShape`/`outputShape` metadata, and batch/epoch `Flow`s. `DataBatch`
  now carries sample `indices` and a `metadata` map, and supports `slice(range)` over the leading batch
  dimension. All additions have defaults — existing `Dataset` implementations keep compiling; the bundled
  MNIST / Fashion-MNIST / CIFAR-10 loaders support indexed (non-contiguous) batches, are routed through
  the source layer, and unsupported platform targets now fail with a clear
  `DatasetLoaderUnsupportedTargetException`. (PR #785)
- **bf16-native DSL → StableHLO export path.** A model authored in bf16 in the NN DSL now exports
  StableHLO whose weights reach the matmuls as bf16 (required by NPUs that reject fp32 weights). Adds
  `DtypeForwardPropagationPass` (unifies the graph's float dtype end-to-end, coercing float sources when
  `targetFloatDtype` is set), a width-matched `-inf` bit pattern for softmax/attention max-reduce
  identities, valid `stablehlo.convert` function-form emission, and BF16 support in
  `DenseTensorDataFactory.zeros/placeholder`. Verified: a DSL-authored bf16 Moonshine encoder traces to
  all-bf16 StableHLO and compiles to an aarch64 llvm-cpu vmfb. (PRs #788, #791)
- **Pluggable per-phase, per-target compile optimization (`skainet-compile-opt`).** The seam that keeps
  hardware knowledge out of the agnostic compiler core: `CompilePhase { TAPE, DAG, STABLE_HLO }`,
  `TargetOptimizer` (per-phase pass provider), the `TargetOptimizers` registry, and
  `dagPipelineFor(target, corePasses)`. The StableHLO emitter additionally accepts an optional target id
  and `OpGranularityPolicy` (+ `FusedOpAllowList`) so per-target fused-vs-decomposed emission decisions
  are possible; everything defaults to the previous behavior — emission is byte-identical when unused. (PR #791)
- **`KernelProfile` diagnostics (`skainet-backend-cpu`).** Always-on accumulating profiler over the three
  `DefaultCpuOps.matmul` dispatch paths (quant-NEON / fp32-scalar / generic), read via
  `KernelProfile.report()` — used to localize on-device decode cost.
- **Contributor Covenant 3.0 Code of Conduct**, with GitHub private vulnerability reporting as the
  contact channel. (PR #790)

### Performance

- **Native CPU K-quant matmul: 2.07× Q4_K on Cortex-A55.** Two levers, validated against the Panama
  reference and on-board: (1) block-outer / output-row-inner loop order so block-major weight bytes are
  read strictly sequentially (the dominant win on in-order cores — the old order made every weight read
  a cold cache miss), applied to Q4_K, Q5_K, Q6_K, and Q8_0; (2) ggml-style fused Q8 activation
  quantization + int8 dot path (`vdotq_s32` on dotprod targets, scalar fallback otherwise) for Q4_K and
  Q6_K. TinyLlama Q4_K_M on-board decode improved 1.50× end-to-end. Note: the fused int8 path is
  deliberately lossy (activation quantization, ~1–3% worst-case on uniform-random fixtures); parity
  tests gate on aggregate `RMS(error)/RMS(signal) < 0.03` instead of bit-exactness. (PR #787)
- **NEON kernels verified on real aarch64.** Shared `nativeTest` suite now runs the matmul parity tests
  on linuxX64 and linuxArm64 (cross-built binary under the bundled qemu-aarch64, overridable to a real
  board); all fp32/q4k/q5k/q6k/q8_0 kernels pass on QEMU and on a physical Cortex-A55, with objdump
  confirming genuine SIMD (`udot`/`sdot`/`fmla`), not the scalar fallback. (PR #786)

### Fixed

- **LayerNorm normalization computed in f32 regardless of model dtype.** A bf16 variance (sum of many
  bf16 squares) loses enough precision that `sqrt(var + eps)` can produce NaN, and some accelerator
  backends miscompile the low-precision decomposed reduce. Mean/variance/std/divide are now upcast to
  f32 (standard PyTorch/JAX practice); the gamma/beta affine stays in the model dtype. No-op for f32
  models. (PR #791)
- **Rank-0 tensor types emit as `tensor<elem>`.** The StableHLO type mapper unconditionally inserted the
  `x` shape separator, so scalars produced the malformed `tensor<xbf16>` that `iree-compile` rejects. (PR #791)
- **Green `./gradlew build` on macOS hosts.** Refreshed lagging binary-compatibility API dumps (additive
  only) and gated the linuxX64/linuxArm64 Kotlin/Native test-link/run tasks off non-Linux hosts (the
  CMake host build emits Mach-O objects that cannot cross-link into a Linux ELF); klib compilation and
  publishing untouched, the native NEON parity suite still runs on Linux CI / QEMU / hardware. (PR #789)

### Docs

- **Antora docs image consolidated to the offline `markup-antora` build.** One image shared across the
  SKaiNET docs projects: offline Mermaid rendering to inline SVG at build time (no Kroki, no network),
  content-hash diagram caching, rootless-safe, with a build-time Mermaid smoke test. (PR #781)
- Data source URIs and the data loader APIs documented, including a new
  `data-sources-getting-started` tutorial; README reworked to frame StableHLO/MLIR as one of several
  sibling code-generation backends (next to Arduino/C99 and Minerva) lowering the same `ComputeGraph`. (PR #776)

### Dependencies

- Gradle wrapper 9.6.0 → 9.6.1, logback-classic 1.5.36 → 1.5.37, JUnit Jupiter 6.1.0 → 6.1.1,
  kotlinx-io-core 0.9.0 → 0.9.1. (PRs #777–#780)

## [0.33.0] - 2026-06-29

### Added

- **GRU layer (`sk.ainet.lang.nn.Gru`).** SKaiNET's first recurrent layer (issue #217): single-layer,
  unidirectional, batch-first `[B, S, D] -> [B, S, H]`, PyTorch gate order (reset, update, new). Built
  by composing existing primitives (matmul/add/sigmoid/tanh/narrow/concat) **unrolled over the static
  sequence length at trace time** — StableHLO has no loop construct, so any recurrence must unroll. It
  runs eagerly, is trainable through the standard tape, and exports to StableHLO with no dedicated
  converter. Also adds a `gru(hiddenSize) { … }` network-DSL builder. (PR #772)
- **`upsample2d` Bilinear + StableHLO export.** Adds the Bilinear forward (PyTorch coord map, 4-neighbour
  blend) and its autodiff backward, and a traceable StableHLO lowering for **both** Nearest and Bilinear
  (scale is static at trace time, so everything lowers to fixed reshape/broadcast/`dot_general` — no
  runtime index math, no `custom_call`). Unblocks export of resize/FPN-style paths. (PR #771)
- **Seven newly-differentiable ops.** `cos`, `sin`, `tril`, `gather`, `indexSelect`, `unfold`,
  `convTranspose1d` now carry `@Diff` and have backward rules (with finite-difference parity tests):
  trig for RoPE, `gather` for embedding lookup, `tril` for causal masks, the rest structural. (PR #774)
- **KSP-generated autodiff-coverage guard.** The tracing-wrapper processor now emits
  `DifferentiableTensorOpsRules.ruleNames` (the authoritative `@Diff` op set); a unit test asserts the
  execution tape's dispatch covers it, so a differentiable op can no longer ship with a backward rule
  that is never wired. `operators.json` now records `isDifferentiable` (+ optional `diffRuleName`),
  schema-validated. (PR #774)

### Fixed

- **Silent gradient drop for `elu`, `leakyRelu`, `permute`.** These were `@Diff` and had correct
  backward formulas, but had no arm in the execution tape's trace dispatch, so their gradients fell
  through to `null` and were silently discarded. Now wired (and guarded by the coverage test above);
  `permuteBackward` also fixed to decode its `axes` attribute as the traced `List<Int>`. (PR #774)
- **`layerNorm` / `rmsNorm` / `batchNorm` lower to real `stablehlo.reduce`.** The norm converters
  previously emitted non-compilable `reduce_mean` / `reduce_variance` `custom_call`s (export-only); they
  now decompose to real `stablehlo.reduce`, so all three compile and run on stock IREE (llvm-cpu). (PR #769)

### Changed

- **BREAKING: `TensorOps.sin`, `TensorOps.cos`, `TensorOps.convTranspose1d` are now abstract.** They
  previously had default `throw NotImplementedError(...)` bodies; they are abstract so the tracing
  wrapper records them (and they become differentiable/exportable). Any type implementing `TensorOps`
  directly must now override them — both bundled backends (`DefaultCpuOpsBase`, `VoidTensorOps`) already
  do. (PR #774)

## [0.32.4] - 2026-06-26

### Fixed

- **Streaming detokenization preserves word-boundary spaces (`Tokenizer.decodeToken`).** A generation
  loop that decodes one token at a time (`decode(tokenId)`) ran words together (`"the process"` →
  `"theprocess"`): the single-token path delegated to the sequence-level
  `SentencePieceTokenizer.decode(IntArray)`, whose `addSpacePrefix` leading-space strip is only
  correct once per sequence. Adds `Tokenizer.decodeToken(id)` (default = `decode(intArrayOf(id))`) and
  a `SentencePieceTokenizer` override that decodes a single token without the leading strip (llama.cpp
  `token_to_piece` semantics), plus a `decode(ids, stripLeadingSpace)` overload. Every streaming
  consumer now reconstructs spacing correctly.

## [0.32.3] - 2026-06-25

### Added

- **Graph-output pruning for export (`ComputeGraph.prunedToOutputs`).** A traced decoder surfaces
  every leaf tensor (e.g. per-layer intermediates) as a graph output, so a StableHLO/IREE export
  returns the logits plus dozens of dangling tensors — extra `func` returns and dead op subgraphs.
  Adds `OutputDesignatedGraph` (skainet-compile-dag) to override the output set by node id, and
  `ComputeGraph.prunedToOutputs(outputNodeIds)` (skainet-compile-opt) which designates those outputs
  and runs `DeadCodeEliminationPass` so only the nodes feeding them survive. Exporters can now keep
  just the logits. Adds `GraphPruningTest` (commonTest).

### Changed

- **SDPA causal mask uses a large finite fill (`-1e30`) instead of `-inf`.** The attention HLO
  converter's causal path emitted `dense<0xFF800000>` (`-inf`) for the masked-fill select; it now
  emits `-1.000000e+30`, matching `MultiHeadAttention.buildSlidingCausalMask` and avoiding a `-inf`
  splat in the IR (numerically equivalent after softmax). (`AttentionOperationsConverter`)

## [0.32.2] - 2026-06-24

### Added

- **`ExecutionContext.isRecording`.** A default-`false` `isRecording` on the lang-core
  `ExecutionContext` interface, overridden by `GraphExecutionContext` (`currentTape?.isRecording`).
  Lets modules with an eager fast-path that bypasses `ops.*` (e.g. RoPE's raw-array INTERLEAVED
  rotation) detect tracing and emit a graph-traceable `ctx.ops.*` path — so they export to StableHLO
  while keeping the fast path for eager inference — without depending on the compile-dag module.
  Backward-compatible (existing `ExecutionContext` implementations inherit the default). (PR #757)

### Changed

- **Docs:** Antora docs version-currency (dependency snippets → current release) and broken-link
  fixes across all pages (`.md` `link:` → `xref:`, dead `tensorflow.org/xla` → `openxla.org`, dead
  `rfc.md`/`bench-prd.md` references, `ainet-sk` → `SKaiNET-developers`). (PR #758)
- **Dependency:** `ch.qos.logback:logback-classic` 1.5.34 → 1.5.35. (PR #756)

## [0.32.1] - 2026-06-23

### Fixed

- **GroupNorm now compiles on stock IREE.** The 0.32.0 GroupNorm converter lowered mean/variance
  with `stablehlo.custom_call @reduce_mean` / `@reduce_variance` (mirroring LayerNorm), which
  `iree-compile` cannot lower — so a `groupNorm` module exported but failed to compile. It now emits
  real `stablehlo.reduce` (add region) + `divide`, computing variance as `E[x²] − E[x]²` (population,
  ddof=0), exactly like the `sum` / `mean` / `variance` converters. Verified end-to-end through the
  `skainet-iree-conformance` harness: `iree-compile` + `iree-run-module` + numpy validate → PASS
  (`max_abs_err = 1.2e-7`). `GroupNormConverterTest` asserts real reductions and no `custom_call`.
  (PR #754)

## [0.32.0] - 2026-06-22

### Added

- **GroupNorm StableHLO converter.** `NeuralNetOperationsConverter` now lowers `groupNorm`
  (and the `groupNormalization` / `GroupNormalization` / `group_norm` aliases) to real
  `stablehlo.*` ops instead of falling through to the "operation not supported" path. The
  lowering mirrors the LayerNorm/RMSNorm decomposition: reshape `(N, C, *spatial)` to
  `(N, G, M)`, per-group `mean` / `variance`, normalize, reshape back, and apply the optional
  per-channel `scale` / `offset`. Adds `GroupNormConverterTest` (commonTest). (PR #752)
- **SKEEP proposals docs module.** (PR #750)
- **Quantization-process explanation doc** (weights, activations, calibration). (PR #747)

### Changed

- **Dependency bumps:** `com.vanniktech.maven.publish` → 0.37.0 (PR #748),
  `com.networknt:json-schema-validator` → 3.0.5 (PR #749), kotest → 6.2.1 (PR #744),
  Gradle wrapper → 9.6.0 (PR #745), `actions/checkout` → 7 (PR #743).

## [0.31.2] - 2026-06-18

### Added

- **`RowDequantSource` + `ops.gather` row-dequant path.** Adds `RowDequantSource` (a `TensorData` marker,
  `dequantRow(rowIdx): FloatArray`) to `skainet-lang-core`, and teaches `DefaultCpuOps.gather` to use it:
  when the gathered table implements `RowDequantSource`, only the rows actually touched are dequantised
  (each unique row once, cached) instead of the generic element path — which calls `get()` (unsupported on
  such tensors) and would otherwise force a full FP32 materialise of the table. The table declares logical
  dtype `FP32`, so `gather` returns FP32 with no typing change. This lets a packed/oversized embedding (a
  Q-quantised `token_embd`) stay packed and be looked up via `ops.gather` directly — generalising the
  per-row-dequant trick out of the model layer. Adds `GatherRowDequantTest` (commonTest). (PR #741)

## [0.31.0] - 2026-06-15

### Fixed

- **`ops.transpose` now lazily handles every packed matmul dtype.** The CPU backend's 2-D transpose rewraps the packed bytes with a flipped shape (a metadata-only "lazy transpose") for the K-series (Q4_K/Q5_K/Q6_K) and Q5_0/Q5_1, but **Q8_0 and Q4_0 fell through to the generic FP32 path**, which cast the Byte-backed buffer to Float and threw `ClassCastException`. Added the `Q8_0TensorData` and `Q4_0TensorData` cases so a packed Q8_0/Q4_0 matmul weight (e.g. a model's tied Q8_0 `lm_head`) survives `linearProject`'s `matmul(x, ops.transpose(W))` and dispatches to its packed kernel instead of crashing — `ops.transpose` now covers the full `chooseQuantizedMatmulHeap` dispatch set (Q4_K/Q5_K/Q6_K/Q5_0/Q5_1/Q8_0/Q4_0). Adds `transpose_preserves_every_packed_quant_type` (commonTest, jvm + linuxX64) as a regression guard. (PR #736, #737)

### Changed

- **Bumped `com.networknt:json-schema-validator` to 3.0.4.** (PR #733)

## [0.30.0] - 2026-06-13

### Added

- **First-class Q5_K packed matmul.** New `TensorEncoding.Q5_K`, `Q5_KTensorData` / `Q5_KBlockTensorData` (256-element / 176-byte super-blocks with the `qh` 5th-bit plane), and a `Q5KMatmulKernel` SPI. Implementations: scalar reference (commonMain → Kotlin/Native, JS, Wasm), JVM Panama Vector, and native-C (FFM). Wired into `DefaultCpuOps` packed-quant matmul dispatch + lazy transpose, registered via `KernelRegistry`, and added to the GGUF `StreamingGgufParametersLoader` (Q5_K + Q6_K packed branches). Q5_K weights stay packed and dequantize inside the matmul, matching the existing Q4_K/Q6_K path. (PR #734)
- **ARM NEON kernels for the native CPU backend.** Hand-written NEON paths for `fp32`, `q8_0`, `q4k`, and `q5k` matmul (shared `skainet_simd.h`), behind `#if __ARM_NEON` so x86 keeps its `-O3 -ffast-math` auto-vectorized scalar path. The native CMake build adds an aarch64 branch (`-march=armv8.2-a+fp16+dotprod`; no `+i8mm` — Cortex-A55 lacks it) and an opt-in `-PcrossArm64` cross-compile with a toolchain file. (PR #734)
- **Kotlin/Native consumption of the C kernels via cinterop.** `skainet-backend-native-cpu` now builds a static archive (`libskainet_kernels.a`) alongside the shared lib and adds `linuxX64` + `linuxArm64` targets with a cinterop `.def`, shared `nativeMain` `NativeKn*MatmulKernel` wrappers, and a `NativeKnKernelProvider` (+ `installNativeKernels()`). On-device Kotlin/Native binaries can now reach the same hand-tuned C/NEON kernels the JVM uses via FFM. (PR #734)

## [0.29.1] - 2026-06-09

### Fixed

- **`sk.ainet.core:skainet-compile-minerva` is now published to Maven Central.** The new Minerva export module (shipped in 0.29.0) applies the publish plugin and was auto-included in the BOM, but it lacked the per-module `gradle.properties` (`POM_ARTIFACT_ID` / `POM_NAME`) that every other published module carries, so its publication had no POM name and never made it to Maven Central. Added the module's `gradle.properties`; the artifact now publishes alongside the rest of the engine.

## [0.29.0] - 2026-06-09

### Added

- **Minerva secure-MCU export module.** A new export pipeline that takes a SKaiNET model all the way to a secure microcontroller project bundle, built up in phases:
  - **Shared graph-export contracts.** A backend-agnostic export contract layer (`feat(export)`), with a StableHLO graph-export adapter that exposes the existing HLO path through those contracts. (#697, #698, #702)
  - **Module API scaffold.** The Minerva export module's public API surface. (#700)
  - **Phase-one graph-compatibility validation.** Validates that a graph is exportable before any lowering work begins. (#701)
  - **IR lowering + npz compiler input.** Compatible graphs lower to Minerva IR (#704) and emit an `.npz` compiler input (#706).
  - **Compiler packager + host verification.** A packager flow that drives libminerva packaging, plus a host-verification flow and runtime-verification profile that prove the exported bundle on the host before it ships. (#706, #712, #714, #716, #717, #721, #725)
  - **Manifest fingerprinting.** Generated manifest artifacts are fingerprinted so bundle contents are tamper-evident. (#724)
  - **Runnable sample + examples + docs.** A runnable sample task and runner, secure MCU export examples, an ONNX export workflow, getting-started / explanation pages, and model-source guidance. (#707, #712, #719, #725)
- **Packed-quant matmul kernels with Kotlin/Native parity.** Q5_0, Q5_1, Q4_K, and Q6_K gain matmul support across the provider stack:
  - **commonMain scalar kernels + SPI** for Q5_1/Q5_0/Q4_K/Q6_K, giving Kotlin/Native parity with the JVM path. (#710, #711)
  - **Packed-quant matmul dispatch in `DefaultCpuOpsBase`** so the packed-quant path is selected on Native, not just JVM. (#709, #711)
  - **Panama Vector (JVM SIMD) kernels** for Q5_1/Q5_0 (#709) and Q6_K, with Q6_K routed via the `KernelRegistry`. (#715, #720)
  - **Packed Q5_1/Q5_0 matmul kernels + lazy transpose** on the CPU backend. (#709)

### Changed

- **Kernel × platform support matrix is auto-generated and CI-gated.** The kernel/platform support matrix is now rendered through the build-logic → Antora pipeline (the same pipeline as the ops docs) from a CI-gated source, so it can't drift from the code. (#716, #724)

### Docs

- Refreshed the architecture kernel-provider section (native FFM ships; link matrix). (#726)
- Eager-execution backends & kernels mindmap, refreshed after Panama Q6_K. (#720, #723)

### Tests

- Hardened CI browser tests against launch flakiness under `allTests` (Karma) and raised the Mocha timeout 10s → 60s for the micrograd demo. (#703, #705)

### Dependencies

- Bumped `io.github.optimumcode:json-schema-validator` to 0.5.5. (#713)

## [0.28.1] - 2026-06-06

### Fixed

- **DAG-DSL StableHLO export now compiles end-to-end with IREE for the full conformance suite (7/7 models, 27/27 ops).** Shape-changing ops declared a result/return type inferred from operand-0 instead of the op's real output, so `iree-compile` rejected the modules (*"inferred shape … is incompatible with return type"*). `DagBuilder.inferDagOutputSpecs` now computes the correct output spec for:
  - **`reshape`/`view`** — reads the target shape from the op's `newShape` parameter (a `Shape`, which the converter's `as? List<Int>` had missed). (#673, PR #674)
  - **`matmul`/`dot`/`mm`/`bmm`** — `(…, M, K) @ (…, K, N) → (…, M, N)` instead of echoing operand-0. (#673, PR #674)
  - **`concatenate`** — the corrected summed-axis extent propagates to the consumers and the `func.func` return type, not just the op line. (#673, PR #674)
  - **`conv1d`** — windowed `(N, Cout, Lout)`; `conv2d` already inferred via `Conv2dOperation`, but `conv1d` was a `GenericOperation`. (#675, PR #676)
  - **`gather`** — `table[:axis] ⊕ indices.shape ⊕ table[axis+1:]`. (#675, PR #676)
  - **`maxpool2d`/`avgpool2d`** — windowed `(N, C, Hout, Wout)`. (#675, PR #676)
  - **`flatten`** — collapses `[startDim..endDim]` while preserving the leading batch dim (it was collapsing everything to rank-1, breaking the dense layer in mnist-cnn). (#675, PR #676)
- **`reduce_window` is emitted in IREE's parseable generic region form.** Pooling previously used the pretty `… applies <op> over window …` form, which IREE rejects (*"has no custom assembly form"*). Now emits `"stablehlo.reduce_window"(…) ({ ^bb0(…): … })` with full NCHW-rank window attributes; average pooling's divisor is splatted to the output type (was a scalar-vs-tensor mismatch). (#675, PR #676)
- **`MlirValidator` understands region block arguments.** It now registers `^bb0(%a, %b)` block-argument SSA definitions and every `%x =` result on a line, so single-line region ops (e.g. `reduce_window`) validate. (PR #676)

### Changed

- Regenerated the JVM binary-compatibility baselines (`apiDump`) to match the public API exposed since 0.27.0 (`AttentionOperationsConverter`, multi-output port helpers, `KClass` dtype constructors).

## [0.28.0] - 2026-06-06

### Fixed

- **`reshape` whose target shape lives only in an operation parameter now lowers.** `ShapeOperationsConverter.convertReshape` previously depended on a declared output `TensorSpec`; a graph that carried the target only in the op's `outputShape` / `shape` / `newShape` parameter produced an empty/untyped module. It now reads the parameter and synthesizes a typed `stablehlo.reshape` result. (Issue #666, PR #670)
- **Multi-input `concatenate` sums the operands' extents on the concatenated axis.** `convertConcat` echoed operand-0's extent into the result type, so e.g. concatenating `1×1×8×8 + 1×4×8×8 + 1×1×8×8` on dim 1 emitted `tensor<1x1x8x8xf32>` instead of `tensor<1x6x8x8xf32>`. The result type now sums the axis across all operands. (Issue #667, PR #670)
- **StableHLO DSL export for constants and reductions.** DAG constants are inlined (baked) into the emitted module rather than externalized as function arguments, and a reduction drops the reduced dimension. (Issue #663, PR #664)
- **`HloGenerator` forward-pass tracing records ops.** The sample input is bound to the tape execution context and external inputs are synthesized (`toComputeGraph(synthesizeExternalInputs = true, …)`), so tracing a `Model` emits real StableHLO ops instead of a structure-only module. (Issue #668)

### Added

- **Non-JVM image runtime support.** Image and data-transform modules are scoped to their supported KMP targets, with a non-JVM image runtime implementation so the image/data-transform APIs build honestly across targets. (PR #671)

## [0.27.0] - 2026-06-04

### Added

- **Full gemma3 network lowers to StableHLO with zero gaps.** A batch of new core converters closes every remaining op gap on the Kotlin DSL → MLIR StableHLO path, so a complete gemma3 graph traces and lowers end-to-end (verified by `GemmaTraceTest` over the composite build: 140 nodes → 255 lines, 0 unsupported, 0 arity errors):
  - **`scaledDotProductAttention` converter** (`AttentionOperationsConverter`) — lowers the atomic SDPA op to the standard StableHLO subgraph: `scores = Q·Kᵀ` (`dot_general`, contract `head_dim`), `* scale` (arg or `1/sqrt(head_dim)`), numerically stable softmax over key length, `out = attn·V`. Batched `[..,S,D]` with all leading dims as batching dims. SDPA is a core `TensorOps` op, so its converter lives in core.
  - **Causal mask for SDPA** — when the node's `causal` attr is set, emits an additive `-inf` mask before softmax (`iota`/`compare GE`/`select`) so each query attends only to keys at or before it. Validated EXACT against a NumPy causal reference and accepted by `iree-compile`.
  - **Explicit SDPA mask operand** — `AttentionOperationsConverter` now consumes `operands[3]` (the additive mask), broadcasting it trailing-aligned to the scores shape before softmax. Fixes gemma sliding-window layers (`causal=false` + explicit causal+window mask) that previously exported unmasked and attended to future tokens. (PR #661)
  - **`permute` + `narrow` converters** — `permute` registered as an arbitrary-axis alias routed to the existing transpose path; `narrow` slicing support.
  - **Multi-output converter support + `split` converter** — per-`(nodeId, outputPort)` SSA naming in `ConversionContext`, operand resolution that walks incoming edges by `destinationInputIndex` and resolves each by the edge's `sourceOutputIndex`, and `split`/`chunk` lowering to N `stablehlo.slice` ops each registered on its own output port (lowers the RoPE `split` gap).
- **Boxing-free `FloatArray` weight externalization for `.irpa` baking.** `finalize()` now stores resolved weights as the primitive `FloatArray` instead of `.toList()` (boxing a real LLM weight — e.g. a 262153×640 embedding → ~2.7 GB `List<Float>` — OOMed the trace). `ConstantOperationsConverter` externalizes `FloatArray` directly (`floatArrayToLittleEndianBytes` + `tryMaterializeExternalFloats`, inlining small/`InlineAlways` tensors via `asList()`), and `IrpaWriter` writes byte ranges in one shot. With this, the real Gemma-270M function bakes: 1 func arg (tokens) + 360 weights externalized to `util.global #flow.parameter.named`.
- **DSL prescribes element dtype for placeholder weights.** The DSL can now specify the element dtype for placeholder weights during tracing.
- **Numerical validation harness for SDPA lowering.** Dumps a small `scaledDotProductAttention` StableHLO graph; `iree-compile` + `iree-run-module` output matches a NumPy reference exactly to 5 decimals, confirming the attention converter is numerically correct, not just structurally valid.

### Fixed

- **IREE-valid StableHLO syntax — full gemma3 compiles to `vmfb`.** Aligned converter emission to what `iree-compile`'s StableHLO parser accepts (verified by compiling the full gemma3 graph end-to-end): `gather` uses the generic MLIR form, `slice`/`narrow`/`split` use the canonical bracket form via a shared `sliceLine()` helper, `concatenate` emits the full functional type, and batch matmul derives batch dims as `min(lhsRank,rhsRank)-2` (fixes 3D-activation @ 2D-weight Linear projections). Result: SKaiNET gemma3 DSL → StableHLO → `iree-compile` (llvm-cpu; +neon aarch64) → `vmfb` for both host x64 and aarch64 targets.
- **`VoidTensorOps.gather` output shape for multi-dim indices.** The void/tracing gather collapsed the gathered axis to `indices.shape[0]`, so a `[vocab,emb]` table with `[batch,seq]` indices traced to `[batch,emb]` instead of `[batch,seq,emb]` (breaking the embedding's downstream reshape during weight-free tracing). Now replaces the axis with the full indices shape, matching `DefaultCpuOps.gather`, unblocking tracing of full transformer (gemma3) graphs.

## [0.26.0] - 2026-05-30

### Added

- **Q4_0 promoted to a first-class quantized format.** The older GGML 4-bit format (18 bytes / 32 elements) was previously a JVM/MemSegment-only, GGUF-only side-path; it is now wired across the full provider stack mirroring Q8_0 / Q4_K:
  - commonMain heap `Q4_0TensorData` / `Q4_0BlockTensorData` (+ `TensorEncoding.Q4_0`) so any loader can produce it and non-JVM targets can use it.
  - `Q4_0MatmulKernel` SPI + `KernelProvider.matmulQ4_0()`, with **scalar** (commonMain), **Panama Vector** (JVM SIMD), and **native FFM** (`skainet_q4_0_matmul`) implementations selected via `KernelRegistry.bestAvailable()` (native → Panama → scalar). `DefaultCpuOpsJvm.chooseQuantizedMatmul` gains an `is Q4_0TensorData` branch.
  - `Q4_0Quantizer` (FP32 → Q4_0) — the produce side was missing, so dense weights from any source (SafeTensors / JSON / in-memory) can now be quantized to canonical ggml Q4_0 without going through GGUF.
  - All Q4_0 paths use the canonical ggml **split** nibble layout (low nibbles → elements 0..15, high → 16..31, `(code - 8) * d`). (PRs #648, #649, #650, #651)
- **`tanh` as a first-class `TensorOps` activation primitive.** Promotes `tanh` from a default `NotImplementedError` stub to a fully wired `@Diff @ActivationDsl` primitive, eliminating the `2*sigmoid(2x)-1` polyfill that downstream consumers were forced to re-derive. Wires the standard six-layer pattern: `TensorOps` interface, `TanhOperation` class, `Tensor.tanh()` extension, CPU backend, recording decorator, and autograd backward (`1 - output^2`). A Karpathy micrograd `demo.ipynb` port lands as an end-to-end training test (moons dataset, `[2,16,16,1]` tanh-MLP, MSE + SGD, held-out accuracy) exercising the primitive through the full DSL + training stack. (Issue #630, PR #631)
- **CPU tensor `convert` op.** Implements the dtype-conversion operation on the CPU backend. (PR #636)

### Changed

- **Recording decorator uses the current recording-stop API internally.** Aligns the internal call site with the current recording API surface. (PR #640)

### Fixed

- **Q4_0 MemSegment matmul layout.** The pre-existing JVM MemSegment Q4_0 kernel (`JvmQuantizedVectorKernels.dotQ4_0BlockMemSeg`) and `Q4MemorySegmentTensorData` used an *interleaved* nibble layout that did not match real GGUF Q4_0 weights (a latent correctness bug; the path was unverified and had no in-repo callers). Reconciled to the canonical split layout so it now agrees with the heap type, the SPI kernels, and `DequantOps.dequantQ4_0FromBytes`. (PR #649)

### Tests

- **Ignored common tests made portable across KMP targets** via the shared `@Ignore` annotation, so the same skip applies consistently on every platform. (PR #638)
- **Ignored-test clarity and BatchNorm coverage.** Clarifies why each ignored test is ignored and enables previously-disabled BatchNorm coverage. (PR #634)

### Build

- **Gradle build-hygiene warnings fixed.** (PR #633)

### CI

- **Feature-PR workflow trigger scope narrowed** to avoid duplicate workflow runs on feature PRs. (PR #645)

### API

- **Resynced binary-compatibility-validator baselines** to current source (the committed `.api` dumps had drifted). (PR #647)

## [0.25.0] - 2026-05-25

### Added

- **BF16 matmul end-to-end.** New `Bf16TensorData` + `Bf16DenseTensorData` (PR #610), opt-in SafeTensors loader policy to keep BF16 native through the loader chain (PR #612), and `DefaultCpuOpsJvm` BF16 dispatch wired against `Bf16TensorData` (PR #614). The matmul kernel itself ships as `Bf16MatmulKernel` with scalar, Panama, and native implementations registered with the `KernelRegistry` SPI (PR #605).
- **Q8_0 matmul end-to-end.** `Q8_0MatmulKernel` with scalar, Panama, and native implementations (PR #606); `DefaultCpuOpsJvm.chooseQuantizedMatmul` now resolves through the `KernelRegistry` SPI so the best-available Q8_0 kernel wins automatically (PR #608).
- **Autograd completeness for `pow`, `log`, and conv/pool/upsample/split.** New `pow` / `powScalar` ops plus a `PowSpecializationPass` (Tier A of #617), `log` / `log2` / `log10` ops (Tier B), backward formulas filled in for `pow` / `log` with the matching dispatch arms (Tier C partial), and the remaining conv / pool / upsample / split backward formulas (Tier C). End-to-end CNN training-step test (Tier D) pins the contract. (PR #618)
- **Hybrid adaptive DSL with optional dtype constraints — RFC implementation.** `DTypeConstraintResolutionPass` registered in the pipeline (W7), a `dtypePolicy(...)` extension on `DagBuilder` (W6), `StreamingGgufParametersLoader.withPolicy(DTypePolicy)` (W0c), a typed `ResolvedComputeGraph` view (W8), and a `toStableHlo(ResolvedComputeGraph)` overload (W9). (Issue #615, PR #616)
- **`@DarcValidated` annotation for operator documentation.** New `sk.ainet.lang.ops.DarcValidated` annotation in `skainet-lang-ksp-annotations`; the KSP `OperatorDocProcessor` threads its values through `operators.json`, and the doc generator renders a `✅ / ⚠ / ✖` badge above each function's signature plus a `Validated` column on the operator coverage matrix. `matmul` is annotated as the worked example. The Antora `Contributing` section gains a dedicated `darc-workflow.adoc` page that defines the workflow and the criteria for setting the annotation; `.github/ISSUE_TEMPLATE/darc_feature_request.md` was aligned (`DEFINE` → `DOCUMENT`, `CONTRIBUTE` → `CODE`). (Issue #627, PR #628)
- **SentencePiece special-token splitter.** New `SpecialTokenSplitter` decorator in `skainet-io-core` plus fixes for SentencePiece HF JSON gaps that previously misrouted control tokens. (PR #595)

### Changed

- **`operator-doc-schema-v1.json` widened to match what the processor actually emits.** Added the `composite` modality, the `inherited` backend status, `version: "unknown"` (mirroring the existing `commit` handling), and the `description` note type. The optional `validated` / `validatedBy` / `validatedOn` / `validatedCommit` / `referencesChecked` fields were added in support of `@DarcValidated`. `validateOperatorSchema` goes from `0/4 valid` to `4/4 valid`. (PR #628)

### Fixed

- **`getInputNodes` ordering.** Now ordered by `destinationInputIndex` so DAG consumers see operands in the right slot. (Issue #620, PR #622)

### Documentation

- **DARC workflow contributor page** (`contributing/darc-workflow.adoc`) — authoritative in-repo definition of Document / Assess / Research / Code, with the operator-doc specialisation. The Antora docs site is now declared the authoritative source over the GitHub wiki. (PR #628)
- **SIMD kernels deep-dive + arc42 architecture page.** How FP32 and quantized Panama Vector kernels are built; how to read the matmul benchmark; the priority-100 native FFM provider plan. (PR #623)
- **Nav split: Using SKaiNET vs Contributing.** The left nav now separates the consumer-facing Tutorials / How-to / Reference / Explanation tree from the maintainer-facing Contributing tree, and Java-specific pages are annotated. (PR #619)
- **Canonical five-minute start path.** New tutorial path that gets a reader from zero to a working SKaiNET call in five minutes; deliberately Kotlin-first to match the framework's primary surface. (PR #624, plus follow-up "keep the start path Kotlin-first")
- **Architecture goal in the README.** Single paragraph stating what SKaiNET is optimising for, so readers landing on the README know the framework's centre of gravity. (PR #625)
- **`native-ffm-plan.adoc` removed from the published docs.** The page read as a PRD/issue, not user-facing reference; content is preserved as an untracked draft for promotion to a real issue. The nine incoming xrefs across six pages were surgically rewritten. (PR #628)

### Dependencies

- Bump `androidGradlePlugin` to 9.2.1 (PR #597).
- Bump `org.jetbrains.kotlinx:kotlinx-benchmark-runtime` to 0.4.17 (PR #599).
- Bump `org.jetbrains.kotlinx:kotlinx-coroutines-core` to 1.11.0 (PR #600).
- Bump Gradle wrapper to 9.5.1 (PR #601).
- Bump `io.ktor:ktor-client-core` to 3.5.0 (PR #602).
- Bump `org.junit.jupiter:junit-jupiter` to 6.1.0 (PR #621).

## [0.23.0] - 2026-05-02

### Added

- **`TensorDataFactory.placeholder(shape, dtype)`** — returns a `TensorData` whose underlying primitive array materializes lazily on first read, instead of allocating a `FloatArray(shape.volume)` eagerly. The default interface implementation falls back to `zeros`, preserving behavior for any custom factory; `DenseTensorDataFactory` overrides with `LazyZeroFloatArrayTensorData` / `LazyZeroIntArrayTensorData`. `ExecutionContext.placeholder(...)` exposes the same path at the `Tensor` level. (PR #588)
- **`PosixPreadRandomAccessSource` for Kotlin/Native** — new public class in `skainet-io-core`'s `nativeMain` source set wrapping POSIX `pread(2)`. `pread` is positional and atomic, so concurrent reads from different positions are safe without locking. Companion `open(path)` returns `null` on open/stat failure to match the JVM `JvmRandomAccessSource.open(...)` behaviour, letting callers cleanly fall back to the legacy sequential reader if needed. Covers `macosArm64`, `linuxX64`, `linuxArm64`, `iosArm64`, `iosSimulatorArm64` — every target in the default `nativeMain` source set on this module. 11 `nativeTest` cases pin the contract (size, partial reads, offset/length variants, EOF/argument validation, idempotent close, missing-file null return). (PR #591)

### Fixed

- **Kotlin/Native consumers couldn't load GGUFs larger than ~2 GiB** — `sk.ainet.io.gguf.createRandomAccessSource(filePath)` on the native target was a placeholder `actual fun … = null`, forcing every K/N caller (`StreamingGGUFReader.open(...)` via the gguf-specific factory, every `*NetworkLoader.fromGguf(...)` path, `LlamaWeightLoader`) to fall through to the legacy reader, which slurps the entire file into a single `ByteArray`. Kotlin arrays cap at `Int.MAX_VALUE` bytes (~2 GiB), so any GGUF over ~1.9 GiB threw `IllegalStateException: Can't create an array of size 2147483648`. Practical impact: macOS / Linux / iOS native builds couldn't open Q8 models above ~1B parameters or Q4 models above ~3B — the JVM target had no such cap because `JvmRandomAccessSource` was already implemented. The `skainet-io-gguf` factory's native actual now delegates to the new `PosixPreadRandomAccessSource` (see *Added* above) and returns the same `null` sentinel on open/stat failure, so existing fall-back code paths remain valid. Verified on macOS arm64 against `Qwen3-1.7B-Q8_0.gguf` (~1.8 GiB), which previously OOMed at construction time. (Issue #589, PR #591)
- **DSL eagerly allocated zero tensors for every Linear / Conv1d / Conv2d, OOMing real-model loaders** — `NetworkBuilder.kt`'s `createLinear`, `DenseImpl`, `Conv1dImpl`, and `Conv2dImpl` paths called `tensorDataFactory.zeros<T, V>(shape, kClass)` eagerly to satisfy each module's constructor whenever the user had not provided initial weights or bias. Downstream loaders always build the network first and only then substitute weights via `WeightMapper.applyWeights`, so the eager zeros were always immediately discarded — but they determined the JVM's peak heap footprint. For `unsloth/Apertus-8B-Instruct-2509-GGUF` (Q4_K_S, 4.7 GB on disk) that was ~27 GB of FP32 zeros allocated and thrown away. Switched every eager-init call site to the new `placeholder(...)` API; the lazy fires only if a caller actually reads the tensor, which never happens on the substitution path because `parameter.value =` swaps the entire `Tensor`. Verified against the real Apertus-8B Q4_K_S GGUF: `ApertusNetworkLoader.fromGguf().load<FP32, Float>(ctx)` now succeeds in 12 GB heap (previously OOMed at 12 GB), constructs all 35 top-level modules in 13 s. Same fix benefits Gemma / Llama / Qwen / Voxtral DSL paths transparently. (Issue #587, PR #588)

## [0.22.2] - 2026-05-02

### Fixed

- **`skainet-bom` published at the wrong Maven coordinates** — the umbrella BOM was being emitted as `sk.ainet.core:skainet-bom` because the engine-wide `GROUP=sk.ainet.core` from the root `gradle.properties` clobbered the per-module `group = "sk.ainet"` override picked up by `vanniktech.maven.publish`. Downstream BOMs (e.g. `sk.ainet.transformers:skainet-transformers-bom`) import this with `<groupId>sk.ainet</groupId>`, so they were unresolvable from a fresh `mavenCentral()`-only project. Fix uses vanniktech's explicit `mavenPublishing { coordinates("sk.ainet", "skainet-bom", VERSION_NAME) }` so the BOM publishes at `sk.ainet:skainet-bom:0.22.2`. `validate-published-poms.sh` extended to assert the BOM landed at the expected path so the regression cannot ship again. (Issue #584)
- **`GgufModelMetadata` silently dropped `UInt`/`ULong` numeric fields** — modern GGUFs (recent llama.cpp converters) store dimensions and counts as `uint32`, which the reader preserves as Kotlin `UInt`. Kotlin's unsigned types do **not** extend `kotlin.Number`, so the previous private `(value as? Number)?.toInt()` helper returned `null` for every `UInt`/`ULong` field. Result: `contextLength`, `embeddingLength`, `layerCount`, `headCount`, `vocabSize` (fallback), `bosTokenId`, and `eosTokenId` all came back `null` on real-world GGUFs and downstream loaders fell back to defaults (e.g. `blockCount=0` → zero-layer transformer). New public file `GgufFieldAccessors.kt` exposes `Map<String, Any?>` extensions (`getInt`/`getLong`/`getString`/`getIntList`/`getStringList`) covering every signed and unsigned integer type the reader can emit, plus the matching primitive arrays for the list variant. `GgufModelMetadata.from()` now routes through these public accessors; the buggy private helpers are deleted. New `GgufModelMetadataUnsignedTest` pins the contract. Non-breaking — only adds public API and fixes existing methods to return correct values. (Issue #585)

## [0.22.1] - 2026-04-30

### Added

- **`StreamingShardedSafeTensorsReader.loadTensorStorageMapped`** — by-name and by-`ShardedTensorInfo` overloads that mirror the existing single-file `StreamingSafeTensorsReader.loadTensorStorageMapped(tensor, filePath)`. Both return a `TensorStorage` whose `BufferHandle.FileBacked` references the resolved shard file's tensor byte range, enabling zero-copy / memory-mapped reads of tensors that exceed the 2 GB JVM `ByteArray` limit. The new methods delegate internally to the per-shard reader; callers don't need to know which physical shard contains a given tensor. Unblocks downstream consumers (e.g. SKaiNET-transformers' Gemma 4 PLE token-embedding table at ~4.7 GB BF16 on E2B) that previously rolled their own `FileChannel.map`. (PR #582)

## [0.22.0] - 2026-04-30

### Added

#### Native (FFM) CPU kernel provider — M5 milestone closed

This release closes milestone M5 of the JVM inference performance roadmap with a priority-100 native kernel provider that wraps a bundled C shared library via Java's Foreign Function & Memory API. Plugs into the existing `KernelProvider` SPI so `KernelRegistry.bestAvailable()` automatically routes Q4_K and FP32 matmul through native when the lib loads, falling back cleanly to the priority-50 Panama Vector kernels otherwise.

- **`skainet-backend-native-cpu` module** — new JVM-only KMP module wrapping a CMake-built shared library (`libskainet_kernels.{so,dylib,dll}`). Bundled into the JAR resources at `native/<os>-<arch>/`, extracted at runtime to a process-scoped temp dir, loaded via `System.load`, and accessed via `Linker.nativeLinker().downcallHandle(...)`. ServiceLoader auto-registers `NativeKernelProviderFactory` via `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider`. (PR #571)
- **Native Q4_K matmul** — single-source scalar C kernel (`-O3 -ffast-math -funroll-loops`); the inner 32-iteration loop auto-vectorizes cleanly into `vfmadd231ps` (AVX2) / `fmla` (NEON). Mirrors `PanamaVectorQ4KMatmulKernel` byte-for-byte on the canonical ggml super-block layout (256 elements / 144 bytes, FP16 d/dMin, 12-byte `get_scale_min_k4` packed sub-scales, 128 bytes of strided 4-bit codes, lazy-`dmin` accumulation). Microbench (Linux x86_64, JDK 21.0.10): **5.87× / 4.71× / 4.17× faster than Panama Vector at 1024² / 2048² / 4096² Q4_K matmul shapes** — single-threaded native beating Panama's `parallelChunks` multi-threaded path on every measured shape. Numerical parity vs Panama within `1e-4` relative tolerance. (PR #572)
- **`Q4KMemSegMatmulKernel` SPI sibling + zero-copy native variant** — JVM-only sibling kernel interface in `skainet-backend-api/jvmMain` taking weights as `MemorySegment` instead of `ByteArray`, plus a JVM-only `MemSegKernelProvider` provider interface that providers can implement alongside `KernelProvider` for the smart-cast lookup pattern at the call site. Reuses the same C symbol as the heap-input kernel — the bytes just don't round-trip through the JVM heap. **+20% wall-clock at 4096²** vs the heap-copy path (9 MB weight transfer eliminated); noise-level at smaller shapes. Bit-identical output to the heap variant. (PR #573)
- **Cross-arch CI matrix** — new `.github/workflows/native-cpu-multiarch.yml` builds and tests the native module on `ubuntu-latest`, `macos-14` (Apple Silicon), and `windows-latest` for every push/PR that touches the native module. Catches portability regressions (linker, alignment, compiler-specific syntax) at PR time rather than after release. C portability tightened: `SKAINET_RESTRICT` macro maps to `__restrict__` on GCC/Clang and `__restrict` on MSVC; CMake grows an MSVC compile-flag branch (`/O2 /fp:fast /W3`) alongside the existing GCC/Clang one. Linux ARM64 was attempted but Kotlin/Native plugin 2.3.21 doesn't support `linux aarch64` as a HOST target ("Unknown host target") — left out for now. (PRs #574, #577)
- **Native FP32 SGEMM** — row-major `C(m,n) = A(m,k) * B(k,n)` with stride support, i-p-j outer-product order so the inner `c[j] += a*b[j]` loop streams two contiguous arrays and auto-vectorizes into FMA. Wired into the existing `matmulFp32()` SPI accessor. Microbench at 256³ / 512³ / 1024³: **1.77× / 1.58× / 1.55× faster than `PanamaVectorMatmulKernel`**. The narrower margin vs Q4_K reflects Panama's already-polished FP32 path (tile-blocking + B-pack + `parallelChunks`); native still wins on every measured shape. Numerical parity within `1e-5 * k` relative tolerance. (PR #575)
- **Multi-arch fat JAR publishing** — `.github/workflows/publish.yml` extended to a two-phase flow: a matrix `build-native` job builds `libskainet_kernels` on each supported host (linux-x86_64, macos-arm64, windows-x86_64), and the `publish` job downloads all three artifacts, stages them into the native module's resources tree, and publishes with every supported arch bundled. Consumers on any of the three arches get a working native path out of the box — no manual side-loading.

#### Module + publishing infrastructure

- **`skainet-backend-native-cpu` registered in BOM** — `skainet-bom` now constrains the new module alongside `skainet-backend-api` and `skainet-backend-cpu`. Consumers depending on the BOM get a constrained version without a separate pin. (PR #576)
- **Publishing config wired** — `vanniktech.mavenPublish` plugin + per-module `gradle.properties` (POM_ARTIFACT_ID + POM_NAME) on the new module. Composite-build consumers (e.g. SKaiNET-transformers via `includeBuild`) substitute the published coordinates with the local project ref through the same path every other SKaiNET module uses. (PR #576)

### Documentation

- **`NativeKernelProvider` consumption kdoc** — covers two gotchas downstream consumers hit on first wiring: (1) the module is JVM-only (FFM has no Native/JS/Wasm equivalents) so KMP consumers must add the dep to `jvmMain.dependencies`, never `commonMain`; (2) `com.gradleup.shadow:9.4.x` `mergeServiceFiles()` silently drops the `NativeKernelProviderFactory` entry when both `skainet-backend-cpu` and `skainet-backend-native-cpu` are on a shadow JAR's classpath — workaround pointer to the `kllama-cli` `doLast` fix in SKaiNET-transformers PR #88. (PR #579)
- **`docs/.../perf/native-ffm-plan.adoc`** — design baseline for the native FFM provider (recovered from the 0.21.0-cycle PRD that was dropped from the repo root and rehomed as asciidoc). Documents module layout, FFM binding pattern, staged delivery, success metrics, and risks.

### Limitations

- **Linux ARM64 native lib is not in the published JAR.** Kotlin/Native plugin 2.3.21 doesn't support `linux aarch64` as a HOST target on the runners GitHub provides, so the cross-arch CI matrix excludes it. Linux ARM64 consumers (Raspberry Pi, AWS Graviton) cleanly fall back to the priority-50 Panama Vector provider — no functional regression, just no native speedup. Re-add when either the Kotlin/Native plugin gains the host or a self-hosted ARM64 runner is wired in.
- **Shadow-jar consumers** using `com.gradleup.shadow:9.4.x` still need a `doLast` workaround to merge the `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` entries — see SKaiNET-transformers PR #88's `kllama-cli`/`skainet-cli` fix for the canonical implementation. Spring Boot apps consuming via Maven (BOOT-INF/lib/) are unaffected.

## [0.21.0] - 2026-04-28

### Added

#### CPU kernel SPI (M5 — JVM Vector half complete)

This release lands the JVM Vector half of milestone M5 from the JVM inference performance roadmap — a pluggable kernel SPI parallel to `BackendProvider`, plus a Panama Vector provider that matches or beats the prior production path on every shape we measure. A native (FFM) priority-100 provider closing the milestone metric is deferred.

- **`KernelProvider` SPI** — `skainet-backend-api` now exposes a `KernelProvider` interface with `name`, `priority`, `isAvailable()`, and per-kernel accessors (`matmulFp32()`, `matmulQ4K()`). `KernelRegistry` does priority-ordered `bestAvailable()` lookup; a JVM-only `KernelServiceLoader.installAll()` auto-discovers providers via `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider`. Manual `register(...)` still works for tests and non-JVM platforms. (PRs #554, #559)
- **`Fp32MatmulKernel` + `PanamaVectorMatmulKernel`** — JDK Vector API implementation using `FloatVector.SPECIES_PREFERRED` + `fma` + `reduceLanes`, cache-blocked with 8×8×128 tiles. `KernelMatmulBench` measures **8.61× / 8.62× / 10.83×** speedup over scalar at 256/512/1024 (JDK 21.0.10, M-series macOS). Within JMH noise of — and often slightly faster than — the prior `JvmVectorKernels.matmulFloatBlocked` production path, so routing introduced no regression. (PRs #557, #558, #560)
- **Production matmul routes through `KernelRegistry`** — `DefaultCpuOpsJvm.matmul` now resolves the FP32 kernel via `KernelRegistry.bestAvailable()` instead of calling `JvmVectorKernels.matmulFloat*` directly. Production `MatmulBench` numbers post-routing match pre-routing within JMH noise. (PR #561)
- **`Q4KMatmulKernel` SPI + SIMD-fused Panama implementation** — Sibling kernel interface in `skainet-backend-api/commonMain`, `KernelProvider.matmulQ4K()` accessor (default-`null` for backwards compat). `PanamaVectorQ4KMatmulKernel` fuses Q4_K dequant inline with the FMA accumulator: a single `ByteVector` load feeds both lo and hi sub-block accumulators per qs slab via AND/LSHR nibble extract → `castShape(B2F)` → FMA, with the lazy-`dmin` correction (`acc += scale·codeSum − offset·inputSum` once per sub-block). `QuantizedMatmulBench` measures 0.07/0.15/0.46 ms at 1024×1024 / 4096×1024 / 4096×4096 (≈30/55/73 GFLOPS — same throughput regime as the FP32 SIMD kernel, meaning fused dequant adds essentially zero cost on top of the FMA). `DefaultCpuOpsJvm.chooseQuantizedMatmul`'s `Q4_KTensorData` branch routes through the SPI with a fall-through to the legacy kernel when no provider resolves. (PR #562)
- **Q4_K MemSeg SIMD** — Same fused-pipeline algorithm applied inline to `JvmQuantizedVectorKernels.matmulF32Q4_KMemSeg` (the path mmap'd weights take). `ByteVector.fromMemorySegment` instead of `ByteVector.fromArray` — no heap copy. (PR #563)
- **Q6_K SIMD dequant** — `dequantQ6_KBlock` replaces its scalar 32-iteration loop with a `ByteVector`-based ql + qh extraction pipeline: per `floatStep`-wide chunk of `l`, loads ql + qh slices, assembles `q1..q4 = (ql nibble) | ((qh slice) << 4) − 32` per lane, multiplies by per-sub-block `d·scale`, stores to four 32-element regions of the scratch FloatArray. (PR #564)
- **Q4_0 partial SIMD** — `dotQ4_0BlockMemSeg` two-stage pattern: scalar byte-pair unpack into a caller-supplied scratch FloatArray (16 byte loads, two nibbles each — half the byte traffic) followed by a `FloatVector` FMA reduction. Closes the last fully-scalar quantized kernel; every quantized format in `JvmQuantizedVectorKernels` (Q4_0, Q4_K, Q4_K MemSeg, Q6_K, Q8_0) is now SIMD'd to some degree. (PR #565)

#### Other

- **`ScratchPool` SPI** — Runtime workspace allocation for transient tensor scratch buffers. Per-runtime size-classed slabs, scoped acquire/release. Closes the framework-side primitive for milestone M1 of the JVM perf roadmap. (PR #550)
- **`TensorOps.permute(axes)`** — Arbitrary-axis permutation (generalizes the existing `transpose` to N-D). (PR #552)

### Fixed

- **Q4_K / Q5_K canonical ggml layout + FP32 MemSeg arena leak** — `Q4_KTensorData` and Q5_K dequant now apply the canonical ggml layout (super-block scale + per-sub-block scaleIdx/minIdx via `get_scale_min_k4` mixing, strided 4-bit codes layout). `MemorySegmentTensorDataFactory` uses `Arena.ofAuto()` for per-op outputs so the matmul / transpose output segments are GC-reclaimable; the prior `ofConfined()` builds leaked tens of MB per matmul, which over a 35-layer Gemma 4 forward pass exhausted the JVM direct-memory cap. Liveness-based freeing of intermediate tensors in `ComputeGraphExecutor`. (PR #556)

## [0.20.0] - 2026-04-24

### Added

#### Quantized matmul (Q4_K / Q6_K on CPU)
- **Q6_K Native Matmul**: New `Q6_KTensorData` / `Q6_KBlockTensorData` in `skainet-lang-core` stores 210-byte ggml Q6_K blocks verbatim (128 `ql` + 64 `qh` + 16 scales + 2 f16 `d`), row-major by default, with a `dequantizeBlock` path matching the `DequantOps` reference line-for-line. `DefaultCpuOpsJvm.chooseQuantizedMatmul` dispatches to a new `JvmQuantizedVectorKernels.matmulQ6_KVec` SIMD kernel (Kotlin Vector API, same `floatSpecies` as the Q4_K / Q8_0 kernels) using a dequant-one-block-to-scratch-then-SIMD-dot pattern. New `TensorEncoding.Q6_K` variant. Unblocks running Gemma 4 E2B Q4_K_M (and any mostly-Q4_K + Q6_K checkpoint) through the DSL path without a ~12 GB FP32 dequant blow-up at load.
- **Q4_K Lazy Shape-Swap Transpose**: `DefaultCpuOpsJvm.transpose(Q4_KTensorData)` now returns a new `Q4_KBlockTensorData` wrapping the *same* packed byte array with swapped shape — mirroring the existing Q4/Q8 MemorySegment lazy-transpose path. `matmulQ4_KVec`'s input-block-major layout produces correct values under the swapped shape without any physical data reordering, so `linearProject(x, W)` can run `matmul(x, transpose(Q4_K_W))` without round-tripping through FP32. Validated at the DSL level by `GemmaDslQ4KTest` in the transformers repo (Δ logits = 4.29e-6 vs the FP32 baseline).
- **Q6_K Lazy Transpose**: Same shape-swap specialization extended to `Q6_KTensorData`, enabling the same DSL path for Q6_K weights.
- **Lazy-Transpose Invariant Tests**: New `QuantizedMemSegMatmulTest` cases pin the two load-bearing properties of the Q4_K and Q6_K transpose specializations — (1) shape is swapped; (2) `packedData` is the SAME byte-array reference, not a copy — so the path cannot silently regress to the generic element-wise transpose (which would `ClassCastException` on packed nibbles).

#### StableHLO → IREE compilation
- **SDPA Recording + StableHLO Emission**: `scaledDotProductAttention` is now recorded by `RecordingExecution` (was silently delegating without recording, like `conv1d` before #532) and lowered to StableHLO by `NeuralNetOperationsConverter`. The decomposition is `dot_general(Q, K.T)` (batching dims `[0,1]`, contracting dims `[3]×[3]`) → scale → optional mask → softmax (max-subtract-exp-sum-div) → `dot_general(weights, V)` (contracting dims `[3]×[2]`). New `ScaledDotProductAttentionOperation` in `TensorOperations` with output-shape inference (output shape = query shape). New `SdpaHloExportTest` verifies tape → graph → MLIR with `dot_general`; `TapeAttentionPermuteBugTest` pins a regression around raw array permute producing zero constants. `ShapeOperationsConverter.concatenate` input-type annotation fix. (#543)

### Fixed
- **SDPA Q/K/V Shape Validation**: `scaledDotProductAttention` previously required only rank-4 inputs, so a mismatch in `head_dim` (e.g. Q=512 vs K=256, as seen in real Gemma 4 E2B where mixed-head-dim layers share a KV cache) surfaced as an `ArrayIndexOutOfBoundsException` buried 2000+ lines deep in the dot-product loop. Added `require()` preconditions on matching batch, head count, Q/K head_dim, Q/V head_dim, and K/V `seqKV`, each with a message naming the offending dimensions. New `SDPAShapeValidationTest` (5 cases, `commonTest`) pins the contract.

### Dependencies
- Kotlin: 2.3.20 → 2.3.21 (including JVM toolchain and `plugin.serialization`).
- Android Gradle Plugin: 9.1.1 → 9.2.0.
- `io.ktor:ktor-client-core`: 3.4.2 → 3.4.3.

## [0.19.1] - 2026-04-21

### Fixed

- **Broken POM for `skainet-backend-cpu`**: The 0.19.0 POM for `sk.ainet.core:skainet-backend-cpu-*` declared a runtime dependency on `sk.ainet:skainet-backend-api-jvm:unspecified` — wrong group coordinate and no valid version, because `skainet-backend-api` was not configured to publish and the root `allprojects { group = "sk.ainet" }` disagreed with the `GROUP=sk.ainet.core` used by vanniktech's maven publish plugin. Consumers pulling 0.19.0 hit unresolved-dependency errors. Fixed by:
  - Applying `vanniktech.mavenPublish` and setting `POM_ARTIFACT_ID=skainet-backend-api` on `skainet-backend-api` so it is actually published alongside the BOM entry that already referenced it.
  - Aligning `allprojects { group = "sk.ainet.core" }` with the `GROUP` property and pinning `version` from `VERSION_NAME` so `project(...)` coordinates in generated POMs are consistent.
- **CI guard**: New `verify-published-poms` job publishes to the local Maven repository and fails the build if any generated `.pom` contains `<version>unspecified</version>` or references a project-local group outside `sk.ainet.core`, preventing a regression of this class of coordinate bug.

## [0.19.0] - 2026-04-20

### Added

#### Tokenizers
- **Qwen / GPT-2 Byte-Level BPE Tokenizer**: `QwenByteLevelBpeTokenizer` implements the full GPT-2-style pipeline — byte-to-unicode mapping, GPT-2 pretokenization regex, merge-rank BPE, and atomic special-token splitting. Builds from either GGUF metadata (`fromGgufFields`) or a HuggingFace `tokenizer.json` (`fromTokenizerJson`). Verified against Qwen2.5-0.5B reference token IDs from HuggingFace `transformers`. (#463)
- **LLaMA / SentencePiece Tokenizer**: `SentencePieceTokenizer` implements the llama.cpp SPM pipeline — whitespace escape (`▁`), code-point symbol split, **score-priority** BPE (the SPM rule, opposite of the merge-rank rule used for GPT-2 BPE), and `<0xNN>` byte fallback for unknown characters. Builds from GGUF (`tokenizer.ggml.model == "llama"`) and HuggingFace `tokenizer.json` (`model.type == "Unigram"`). Verified against TinyLlama-1.1B reference token IDs from HuggingFace `transformers`. (#464)
- **`TokenizerFactory` with Per-Architecture Dispatch**: Tokenizer selection is now **per-architecture, not per file format**. `TokenizerFactory.fromGguf(fields)` and `.fromTokenizerJson(json)` inspect `tokenizer.ggml.model` / `model.type` and dispatch to the right implementation — Qwen/GPT-2 → byte-level BPE, LLaMA/Gemma/TinyLlama → SentencePiece — regardless of whether weights come from GGUF or SafeTensors. (#463)
- **`Tokenizer` Interface**: Common surface implemented by `TekkenTokenizer`, `QwenByteLevelBpeTokenizer`, and `SentencePieceTokenizer` (`encode`, `decode`, `vocabSize`, `bosTokenId`, `eosTokenId`).
- **GGUF Tokenizer Metadata**: `GgufModelMetadata` now exposes `tokenizerModel`, `tokenizerTokens`, `tokenizerMerges`, `tokenizerTokenTypes`, `bosTokenId`, and `eosTokenId` so callers can build a tokenizer without re-parsing the raw field map.

#### StableHLO → IREE compilation
- **Whisper Encoder E2E**: Whisper encoder now compiles end-to-end via SKaiNET → StableHLO → IREE.
- **Real StableHLO Lowerings**: `softmax`, `layerNorm`, and `rmsnorm` now lower to real StableHLO ops (reductions, `broadcast_in_dim`, standard ops) instead of `custom_call` stubs. (#467, #479, #480)
- **New Op Converters**: `gather` / `embedding`, and `concat` / `slice` / `cast` StableHLO converters. (#483, #489)
- **Activation Alias**: `silu` / `SiLU` registered as an alias for `swish` in `ActivationOperationsConverter`. (#484)
- **`ConstantMaterializationPolicy`**: Seam for externalizing large weight tensors out of the StableHLO module (enables `.irpa` externalization). (#524)
- **Splat Constant Folding**: Uniform-value tensor constants collapsed to `dense<v>` splat instead of fully materialized arrays. (#522)
- **SSA Value Type Tracking**: Tracks SSA value types so `reshape` emits the operand's declared type, producing valid MLIR. (#521)
- **Tensor Encoding in Output**: `tensor_encoding` comments in StableHLO output and a top-level `skainet.tensor_encodings` module attribute. (#473, #477)

#### IREE `.irpa` weight files
- **`skainet-io-iree-params` Module**: New module with `IrpaWriter` for writing IREE Parameter Archive (`.irpa`) files. Accepts `FileBacked` handles via mmap on JVM / Android for zero-copy weight export. (#523, #525, #528, #529)

#### Backend API
- **`skainet-backend-api` Module**: New module cleanly separating backend contracts; CPU backend now depends on it. (#468)
- **`TensorEncoding` Metadata**: Accessor for `TensorSpec.metadata` and propagation through `TraceToGraphBuilder.finalize`, keeping quantization encoding visible end-to-end. (#469)

#### Java API (0.19.0 surface polish)
- Annotated `StableHloConverterFactory` and `TokenizerFactory` for idiomatic Java call sites. (#400)
- Renamed `TensorSpecEncoding.kt` class for Java callers. (#400)
- Added `skainet-backend-api` to the BOM. (#400)
- New `ReleaseApiJavaTest` covering the 0.19.0 Java surface. (#400)

#### Docs (Antora migration)
- **Antora + Diátaxis**: Migrated docs to Antora with Divio / Diátaxis layout (tutorials, how-tos, reference, explanation). (#494)
- **`skainet-docs-ui` v1.1.1**: Adopted the new theme with Diátaxis card-grid landing page. (#501)
- **Operator Coverage Matrix**: Emit cross-backend Operator Coverage Matrix generated from `TensorOps` surface scan. (#494, #511)
- **Ops Docs**: KDoc `@param` extraction, real version stamps, LaTeX rendering, fixed partials, and dropped void backend. (#511, #513)
- **Dokka API Bundle**: Wired into the Antora site build. (#494)
- **Local Mermaid**: Drop kroki, render Mermaid locally via `mmdc`. (#496)

#### Platform targets
- **`androidNativeArm32`**: Added across core modules. (#503)

### Fixed
- **Byte-Level BPE Broken for Qwen/GPT-2 Models**: Previously there was no GPT-2-style byte-level BPE tokenizer in the repo, and `GgufModelMetadata` ignored `tokenizer.ggml.merges` entirely — so any Qwen / GPT-2 / Mistral-Nemo model encoded text into garbage tokens (byte-level chars instead of merged vocab IDs), blocking chat mode and tool calling. The new `QwenByteLevelBpeTokenizer` + `TokenizerFactory` dispatch fix the issue for both GGUF and SafeTensors sources. (#463)
- **No SentencePiece Path for LLaMA-Family GGUF Models**: `TokenizerFactory` previously threw `UnsupportedTokenizerException` for `tokenizer.ggml.model == "llama"`, leaving LLaMA / TinyLlama / Gemma / Mistral-v0.1 GGUFs untokenizable. The new `SentencePieceTokenizer` closes that gap. (#464)
- **GGUF UInt Fields Silently Dropped**: GGUF UINT32 fields (e.g. `tokenizer.ggml.bos_token_id`) arrive from `StreamingGGUFReader` as `kotlin.UInt`, which is a value class — *not* a subclass of `kotlin.Number` — so a plain `as? Number` cast was returning null. The new `toIntFlexible` helper handles every signed and unsigned numeric type GGUF can produce, restoring the BOS/EOS/UNK ids on the tokenizer builders.
- **Graph Conv Output Shape Inference**: `conv1d` / `conv2d` / `conv3d` operations in graph inference previously produced placeholder output shapes, breaking downstream shape-dependent passes. Graph ops now compute real output shapes. (#536, #537)
- **Conv1d/Conv3d Not Recorded**: `conv1d` and `conv3d` were not routed through the recording decorator, so they disappeared from traced computation graphs. (#532, #533)
- **Static Conv1d HLO Shape Crash**: Conv1d StableHLO lowering crashed when trace attributes were missing; now falls back to `TensorRef` shape / dtype. (#530, #531)
- **Flatten Hardcoded to MNIST Shape**: `NetworkBuilder.flatten()` returned a hardcoded `lastDimension = 1568` (the MNIST CNN value); any other architecture — e.g. a 64-channel CNN over 32×32 inputs — crashed with `ArrayIndexOutOfBoundsException` in the following `dense()` layer. The DSL now tracks per-sample shape through a new `input(IntArray)` overload, `conv1d` / `conv2d` / `conv3d`, `maxPool2d`, `avgPool2d`, and `upsample2d`, reusing the `ConvShapeUtils` arithmetic introduced in #537; `flatten()` reads the tracked shape and honors `startDim` / `endDim`, and `Conv*` layers can auto-infer `inChannels` from the declared input. (#535, #538)
- **StableHLO `transpose` / `dot_general` MLIR Emission**: Fixed malformed MLIR produced by `stablehlo.transpose` and `stablehlo.dot_general` that blocked IREE compilation. (#520)
- **WasmJS / JS / Native Compile**: Replaced JVM-only `putIfAbsent` with a common-stdlib idiom. (#485)
- **Antora Container**: `HOME=/tmp` so Chromium crashpad can launch during Mermaid rendering in CI. (#534)
- **`bundleDokkaIntoSite` CI Permission Failure**: Fixed docs pipeline permission error. (#496)
- **Pandoc Artifacts in Docs**: Stripped pandoc anchors and demoted heading levels in migrated pages. (#496)

### Changed
- **`compile-hlo` Dependencies**: Dropped vestigial `skainet-backend-cpu` dependency from `compile-hlo` jvmMain. (#472)
- **Moved-LLM Docs**: Replaced relocated LLM pages with redirect stubs pointing at the standalone repo. (#499)
- **Maven Group / Version Refs**: Bumped stale version references and fixed Maven group coordinates. (#499)

### Removed
- Stale `TURBOQUANT_ISSUES.md` tracker at the repo root. (#490)

### Dependencies
- agp: 9.1.0 → 9.1.1.
- com.networknt:json-schema-validator: 3.0.1 → 3.0.2.
- org.jetbrains.kotlinx:kotlinx-serialization-json: bumped to 1.11.0.
- actions/checkout: 4 → 6.
- actions/upload-pages-artifact: 3 → 5.
- actions/cache: 4 → 5.
- actions/setup-java: 4 → 5.
- actions/deploy-pages: 4 → 5.
- actions/github-script: 8 → 9.
- docker/build-push-action: 5 → 7.
- docker/setup-buildx-action: 3 → 4.

## [0.18.0] - 2026-04-08

### Added
- **TurboQuant KV-Cache Compression**: Runtime KV-cache compression for LLM inference using rotation-based quantization (Google Research TurboQuant paper). Supports PolarOnly and PolarPlusQjl variants with 2/3/4/8-bit encoding.
  - `TurboQuantCodec`: End-to-end encode/decode pipeline (random rotation, scalar quantization, QJL residual, bit-packing).
  - `TurboQuantKvCacheStore`: Compressed KV cache with per-head TurboQuant blocks and asymmetric K/V policies.
  - `TurboQuantPresets`: Named presets — `safe-lowbit` (Q8_0-K + TQ4-V), `balanced` (TQ4/TQ4), `experimental-max` (TQ3/TQ3).
  - `KvCacheStore.turboQuant("balanced", ...)`: One-line factory for skainet-transformers integration.
  - `CompressedKvAttention`: SDPA bridge with FULL_TILE and RAW_STORAGE dequant strategies.
  - `@KvCache` and `@KvCacheBypass` DSL annotations for declarative KV cache configuration.
  - `KvCacheAnnotationResolver`: Resolve annotations to cache instances.
  - `TurboQuantUsage`: Documented integration guide with compilable examples.
- **Memory Architecture Hardening**: First-class storage and placement abstractions for zero-copy, quantization-preserving tensor management.
  - `TensorStorage`: Runtime descriptor replacing ad-hoc array passing (logical type, physical encoding, buffer ownership, placement).
  - `TensorEncoding`: Sealed hierarchy — `Dense`, `Q4_K`, `Q8_0`, `TernaryPacked`, `TurboQuantPolar`, `TurboQuantPolarQjl`, `Opaque`.
  - `BufferHandle`: Five ownership modes — `Owned`, `Borrowed`, `Aliased`, `FileBacked`, `DeviceResident`.
  - `Placement`: Device/memory-domain intent with fallback policies (`CPU_HEAP`, `MMAP_WEIGHTS`, `GPU_PREFERRED`).
  - `LogicalDType`: Semantic numeric types separate from physical encoding.
  - `PackedBlockStorage`: Unified contract for all packed quantized formats.
  - `MemoryPlanner`, `MemoryTracker`, `ActiveMemoryTracker`: Placement resolution and copy diagnostics.
- **KV-Cache Subsystem**: `KvCacheStore` interface with append-by-token writes, layer/head addressing, eviction, and `DefaultKvCacheStore` (dense FP32 baseline).
- **Quantization-Preserving Loaders**: `StreamingGGUFReader` and `StreamingSafeTensorsReader` produce `TensorStorage` with `FileBacked` or `Borrowed` handles (no forced densification).
  - `StorageAwareSafeTensorsLoader`: Zero-copy file-backed SafeTensors loading.
  - Completed `Quants.kt` port: `byteShapeToQuantShape`, `quantByteSize`, `isBlockQuantized`, `validateQuantizedBytes`.
- **Tekken Tokenizer**: Mistral Tekken (tiktoken-based BPE) tokenizer support.
- **CPU SIMD TurboQuant Kernels**: `JvmTurboQuantKernels` with Java Vector API acceleration for abs-max, quantize, dequantize, and Walsh-Hadamard butterfly.
- **JMH Benchmarks**: TurboQuant encode/decode throughput, bit-packing, rotation, and KV cache append/read benchmarks (`TurboQuantBenchmarks.kt`).
- **Storage Benchmarks**: Dequantization throughput (Q4_K, Q8_0, Ternary), buffer accessor, and TensorData bridge benchmarks (`StorageBenchmarks.kt`).
- **New Ops**: `sin`, `cos`, `tanh`, `convTranspose1d`.
- **New Layers**: `TransposedConv1d`, `Snake` activation, `LayerScale`.

### Changed
- **Streaming GGUF as Default**: `StreamingGGUFReader` is now the recommended GGUF loading path (memory-efficient, supports quantized types).
- **DSL Annotations**: Extended `PlacementAnnotations.kt` with `@KvCache(preset=...)` and `@KvCacheBypass` for TurboQuant configuration.

### Fixed
- **Int Overflow for Large Tensors**: Fixed `StreamingTensorInfo.nBytes` and `StreamingSafeTensorInfo.sizeInBytes` from `Int` to `Long`, preventing silent overflow for tensors > 2 GB. Fixes loading of Gemma 4 E4B and future large models. (#452)
- **Legacy GGUFReader Overflow Guard**: Added explicit overflow check with actionable error message for tensors > 2 GB in the legacy eager loader.

### Dependencies
- io.github.kotest:kotest: 6.1.9 → 6.1.11.
- com.squareup:kotlinpoet: 2.2.0 → 2.3.0.

## [0.17.0] - 2026-03-25

### Added
- **Core Engine Focus**: Refactored the repository to focus on the core `ComputeGraph` framework, compiler, and backends.
- **Standalone Ecosystem**: Extracted high-level LLM and transformer implementations to dedicated repositories ([SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) and [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers)).
- **LLM-as-DSL**: High-level DSL for defining and running LLM architectures within the core `ComputeGraph` framework.
- **ComputeGraphExecutor**: New optimized executor with support for fusion passes and trace-to-DAG bridging.
- **SDPA & Gather**: Implementation of Scaled Dot-Product Attention (SDPA) and `gather`/`indexSelect` ops across backends.
- **EmbeddingAdapter**: Streamlined embedding layer integration for transformer models.

### Changed
- **Optimized LLM execution**: Integrated fusion passes for faster inference on supported backends.
- **Improved Tensor API**: Refined `Tensor` interface and updated `ComputeGraphExecutor` for better type safety and performance.
- **Dependency Cleanups**: Removed stale references to LLM and transformer code already moved to the standalone `skainet-transformers` repository.

### Fixed
- **Embedding Padding**: Fixed `paddingIdx` handling in embedding layers.
- **Concatenation**: Resolved rank-specific issues in tensor concatenation (rank > 1).
- **Compilation**: Fixed various build and compilation errors after module migrations.

## [0.16.0] - 2026-03-08

### Added
- **Deduplicated LLM infrastructure**: unified `KvCache`, `softmax`, `RoPE`, and `sampling` logic across modules for improved maintainability.
- **Updated skainet-bom**: Refactored the Bill of Materials (BOM) to use local `project()` references for better build consistency.

### Changed
- **LLM Module Extraction**: Extracted and moved core LLM modules to the standalone [SKaiNET-LLM](https://github.com/SKaiNET-developers/SKaiNET-LLM) repository to reduce core codebase footprint.
- **Transformer Code Cleanup**: Removed redundant code that has been moved to the [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers) repository.

### Fixed
- **Dependency Graph**: Resolved inverted dependency issues in the LLM infrastructure.

## [0.15.3] - 2026-03-07

### Added
- **System Prompt Support (Java)**: Added `systemPrompt` support to `KLlamaJava` and `KLlamaSession` for prepending system instructions to conversations.
- **Model Module Extraction**: Extracted model-specific code into dedicated `skainet-models` modules for better separation of concerns and maintainability.
- **Enhanced Smoke Tests**: Refactored `smoke-test.sh` to support multiple runners via JSON configuration and improved LLM loading verification.

### Fixed
- **Whisper HLO Generation**: Fixed StableHLO MLIR generation for Whisper models.
- **Compilation**: Fixed various Kotlin/JVM compilation errors.

## [0.14.0] - 2026-03-03

### Added
- **First-Class Java 21+ Support**: Complete Java API surface with `SKaiNET` entry point, `TensorJavaOps`, builder-pattern model definition (`SequentialModelBuilder`), `KLlamaJava`/`KBertJava` facades, `JavaAgentLoop` for tool-calling agents, and `TrainingLoop` builder.
- **Maven BOM**: New `sk.ainet:skainet-bom` artifact for one-line version management across all modules.
- **Java Documentation**: Added [Getting Started](docs/java-getting-started.md), [LLM Inference](docs/java-llm-inference.md), and [Model Training](docs/java-model-training.md) guides.
- **Java 25 Performance Documentation**: Added documentation for JVM CPU backend performance advantages.
- **WasmWasi Target**: Added `wasmWasi` target support across all KMP modules.
- **StableHLO MLIR Streaming API**: New `HloGenerator` public API with generic Model + Tensor interface and streaming MLIR output.
- **ReductionOperationsConverter**: Added support for reduction operations in StableHLO export.
- **JVM Performance (Jlama Techniques)**: MemorySegment-based tensors, SIMD GEMM kernels, paged KV cache, batch attention for prompt prefill, fused QKV projections, and cached quantized weights.
- **Native RandomAccessSource**: POSIX `pread()`-based source for memory-efficient GGUF parsing.
- **MemorySegment Weight Conversion**: New `NATIVE_OPTIMIZED` quant policy and `MemSegWeightConverter` pipeline with Arena lifecycle management.
- **Lazy Transpose**: Added lazy transpose for Q4/Q8 MemorySegment tensors and MemSeg FP32 transpose.
- **Java CLI App**: New Java-based KLlama CLI application.

### Changed
- **Android KMP Plugin Migration**: Migrated Android subprojects to `androidMultiplatformLibrary` plugin for AGP 9 compatibility.
- **Refactored Model Loading**: Extracted shared dequantization, registry, tensor naming, and decoder runtime into reusable components.
- **JDK Requirement Relaxed**: Allow JDK >= 21 instead of requiring exactly JDK 21.
- **Gradle Upgrade**: Updated to Gradle 9.3.1.
- **Kotlin Upgrade**: Bumped Kotlin from 2.2.21 to 2.3.10.
- **Kotlin Compile Testing**: Replaced abandoned `kotlin-compile-testing` with `kctfork` for Kotlin 2.3.0 compatibility.

### Fixed
- **StableHLO MLIR Export**: Fixed MLIR export to produce valid IREE-compilable output.
- **OOM in Dequantization Benchmark**: Fixed out-of-memory in `DEQUANTIZE_TO_FP32` E2E benchmark test.
- **Quantized MatMul**: Fixed block offset calculation in quantized matrix multiplication.
- **CI Stability**: Fixed AAPT2 daemon crashes and improved Android build stability.
- **Documentation CI**: Fixed workflow permissions for PR comments.
- **Deprecated API Usage**: Fixed `createTempDir()` deprecation in data-simple integration tests.

### Dependencies
- com.gradleup.shadow: 9.3.1 → 9.3.2.
- com.fasterxml.jackson.core:jackson-databind: 2.21.0 → 2.21.1.
- ch.qos.logback:logback-classic: 1.5.27 → 1.5.32.
- io.github.kotest:kotest: 6.1.3 → 6.1.4.
- org.jetbrains.kotlinx:kotlinx-io-core: 0.8.2 → 0.9.0.
- com.vanniktech.maven.publish: → 0.36.0.
- org.jetbrains.kotlinx.kover: → 0.9.7.
- actions/setup-node: 4 → 6.
- actions/upload-artifact: 6 → 7.
- actions/download-artifact: 7 → 8.
- junit-platform-launcher added for CI test execution.

### Contributors
Thank you to the following contributors for their work on this release:
- **Dhia Chemingui** ([@dhiaspaner](https://github.com/dhiaspaner)) — Android KMP plugin migration (#385, #386)

## [0.13.0] - 2026-02-12

### Added
- **Tool Calling**: Added support for tool calling in KLlama, including a new `skainet-kllama-agent` module.
- **Gemma 3n Support**: New `skainet-kgemma` module for Google's Gemma 3n E2B multimodal models.
- **Extended SafeTensors Support**: Added SafeTensors weight loading support for both KLlama CLI and Gemma models.
- **HuggingFace Tokenizer**: Initial support for HuggingFace-style tokenizers in Gemma models.

### Changed
- **Named Arguments**: Refactored various internal APIs to use named arguments for better optional parameter support.
- **System Prompt Handling**: Improved system prompt formatting and handling in agentic workflows.

## [0.12.0] - 2026-02-10

### Added
- **BERT Support**: Full support for BERT-based models with `SafeTensors` weight loading.
- **kbert-cli**: New CLI tool for running BERT inference, supporting text encoding and cosine similarity computation.
- **WordPiece Tokenizer**: Implementation of WordPiece tokenizer for BERT models.

## [0.11.0] - 2026-02-08

### Added
- **TinyFoA Support**: Implemented missing operators (`abs`, `sign`, `clamp`, `lt`, `ge`, `narrow`, `pad2d`, `unfold`) to support TinyFoA (AAAI 2025) training pipeline for memory-efficient on-device learning.
- **Multi-platform KLlama**: Added macOS target support for the KLlama runtime.
- **Custom Backends Documentation**: Added detailed guide and examples for injecting custom backends into KLlama.

### Fixed
- Improved robustness of TinyFoA operations with comprehensive unit tests.

## [0.10.1] - 2026-02-01

### Added
- **Benchmarking DSL**: New `BenchmarkDsl` and `BenchmarkRunner` for measuring model performance and latency.
- **Execution Observers**: Added `ExecutionObserver` API with `LatencyExecutionObserver` and `MemorySnapshotObserver` for profiling.
- **New Layers**: Added `RMSNormalization` layer support.
- **KLlama Enhancements**: Improved weight loading and initial support for GPU-accelerated attention (experimental).

### Changed
- Refactored `ExecutionContext` to support execution observers and better phase management.
- Updated KLlama runtime with improved ingestion and benchmarking utilities.

## [0.9.2] - 2026-01-27

### Added
- **Generative AI Section**: New README section with simple code for GGUF text generation.
- **Tokenizer Strategies**: Automatic detection of tokenizer type (SentencePiece, BPE, WordPiece) from GGUF metadata.
- **Improved Token Decoding**: Support for multi-byte UTF-8 character decoding from byte tokens.

### Changed
- **Llama Runtime**: Rewritten `matmulNoBias` for better performance and support for row-major weights.
- **GGUF Loading**: Improved dequantization for Q2_K, Q4_K, Q5_K, and Q6_K formats matching llama.cpp logic.

### Fixed
- **GGUF Storage Order**: Fixed critical bug with column-major storage in GGUF files by implementing proper transposition during loading.
- **Llama Attention**: Fixed missing attention output projection (wo) in the runtime.
- **Tokenizer**: Fixed BOS token handling and multi-byte character reconstruction.

## [0.9.1] - 2026-01-26

### Added
- **SafeTensors Support**: Initial implementation of `skainet-io-safetensors` for reading SafeTensors format.
- **Generalized I/O & Weight Mapping**:
    - New `WeightMapper` and `WeightLoader` APIs for unified model parameter loading across formats.
    - `LoadingProgress` API for tracking model loading state.
    - `GgufModelMetadata` and `OnnxModelMetadata` for better inspection of model files.
- **JVM Performance**: Enhanced `DefaultCpuOpsJvm` with `JvmVectorKernels` for SIMD-accelerated tensor operations using the Java Vector API.
- **Llama Enhancements**:
    - Added `GGUFTokenizer` for better text processing.
    - Improved `LlamaIngestion` and ingestion pipelines.

### Changed
- **Improved GGUF/ONNX Loading**: Robust weight loading and metadata parsing for GGUF and ONNX models.
- **Streamlined CLI**: Removed unfinished CLI samples and reorganized `skainet-tensor-tools`.
- **Documentation Cleanup**: Removed outdated technical docs and consolidated architecture information.

### Fixed
- Improved robustness of GGUF and ONNX streaming readers.
- Fixed various issues in WASM/JS weight parsing.

## [0.8.3] - 2026-01-18

### Changed
- Updated version to 0.8.3.

## [0.8.2] - 2026-01-18

### Added
- **KLlama (Llama 2 port)**: Initial version ported from `llama2-kmp`, supporting GGUF models.
- **GGUF Enhancements**:
    - Support for `mmap` for zero-copy GGUF tensor loading.
    - Embedded tokenizer support in GGUF.
    - New quantization formats: `Q8_0`, `Q4_K`, and BitNet/Ternary support (`TQ1_0`, `TQ2_0`).
    - Improved loading and bug fixes for quantization and mapping.
    - Added `int64` support for GGUF.
    - Improved GGUF metadata loading.
- **Streaming Support**: Added streaming support for GGUF and ONNX models.
- **Advanced Operations**:
    - New activations: `LeakyReLU`, `ELU`.
    - New pooling: `AvgPool2d`.
    - New convolutions: `Conv1d`, `Conv3d`.
- **Optimizers & Training**:
    - Added `Adam` and `AdamW` optimizers.
    - Comprehensive loss function library.
    - New `Metric` interface with `Accuracy` implementation.
    - KSP-based DSL generator for Network activations.
- **Data & Datasets**:
    - Support for `CIFAR-10` and `Fashion-MNIST` datasets.
    - New `Data Transform API` and `Image Transform DSL`.
- **Testing & Documentation**:
    - `skainet-test-groundtruth` module for validation against PyTorch.
    - Integration tests for quantized inference and `KvCache`.
    - Shadow JAR support for JVM fat JAR builds.
    - New documentation for testing architecture with Mermaid diagrams.
- **WASM/JS**: Initial version of a simple WASM/JS sample.

### Changed
- Simplified model support to **GGUF-only** (removed legacy Karpathy `.bin` format support).
- Improved KLlama loading and robustness.
- Updated roadmap with Phase 1 completion and multi-backend storage abstraction plans.
- Improved I/O system and overall robustness.

### Fixed
- Fixed various bugs in quantization and memory mapping.
- Resolved compilation errors and failing tests in CIFAR-10 support.
- Fixed KSP and TracingWrapperProcessor tests to match updated log messages.
- Fixed GGUF metadata loading issues.

## [0.8.1] - 2026-01-18
- Initial release of 0.8.x series.

## [0.7.1] - 2026-01-14

### Added
- **Sine Approximation CLI** (`skainet-sine-approx-cli`) as a new example application for training models.
- `TapeRecordingStrategy` to handle different recording behaviors for prediction and backpropagation.
- Comprehensive E2E tests for training sine wave approximations.
- New documentation: `autograd-basic.md` explaining the autograd engine.

### Changed
- Refined `Linear`, `Flatten`, `Input` modules and `relu` activation to better support gradient tracking and context propagation.
- Improved `DefaultExecutionTape` and `DefaultGraphExecutionContext` for more robust computation tracing.
- Optimized internal `OpSink` and `TraceSession` handling.

### Fixed
- Infinite loop error during backpropagation tracing by implementing specialized tape recording strategies.
- Context mismatch errors in backpropagation tracing.
- Broken testing in the sinus sample application.

## [0.7.0] - 2026-01-14

### Added
- Initial **Autograd** engine (`DefaultGradientTape`) for automatic differentiation and reverse-mode gradients.
- **Optimizer API** with `SgdOptimizer` implementation for training neural networks.
- **Loss functions** module including `MSELoss` and `CrossEntropyLoss` with configurable reductions (MEAN, SUM, NONE).
- **Training DSL** and helper utilities for building training loops (`trainStep`, `evaluateLoss`).
- Improved **Graph DSL** with better context propagation and support for recording computation traces.

### Changed
- Updated dependency versions and refined internal execution context APIs to support gradient tracking.
- Refactored `skainet-compile-dag` to support autograd and graph inversion.

## [0.6.0] - 2025-12-31

### Added
- StableHLO implementation and E2E CLI app for compiling models to CUDA via IREE.
- `ArduinoCodegen` for exporting models to standalone C99 code with static memory allocation, optimized for Arduino.
- KSP-based generation of `TracingOps` for automated recording pipeline updates.
- Initial implementation of `skainet-compile-hlo` for high-level optimization.

### Changed
- Improved CUDA backend strategy and IREE integration.
- Optimized long-running property tests for C code generation.
- Refactored `TracingTensorOps` to use execution context for code generation.

## [0.5.1] - 2025-12-26

### Added
- Common I/O abstraction with `ModelReader` and `TensorInfo` in `skainet-io-core` for unified model loading.
- Efficient memory handling with non-copying `slice` views in `MemoryChunk`.
- Unified `skainet-tensor-tools` CLI combining ONNX and GGUF utilities.
- `OnnxStatsCli` tool for analyzing ONNX model parameters and structure.

### Changed
- Migrated project to `SKaiNET-developers` organization; updated repository URLs and deployment configurations.
- Standardized artifact naming in documentation (e.g., `SKaiNET-lang-core`).
- Improved `GGUFReader` with better alignment parsing and tensor data handling.
- Optimized test infrastructure: increased heap size to 8GB for large model tests and added `ReadmeSnippetsTest` for documentation verification.

### Removed
- Legacy standalone applications and tools: `skainet-KGPChat`, `skainet-mnist`, and separate ONNX/GGUF tool modules.

## [0.5.0] - 2025-12-06

### Added
- ONNX import module (`skainet-io-onnx`) with pbandk-generated proto surface, loader utilities, and importer that maps ONNX graphs into SKaiNET compute graphs, plus doc and tests.
- CLI tooling: `skainet-onnx-tools` to export ONNX initializers to JSON and `skainet-onnx-detect` CLI to run YOLO detections from ONNX weights.
- YOLOv8 model upgrades: depth/width scaling, decoupled heads with DFL projection, class-name parsing, and detection helpers to align with ONNX exports.
- Image IO module now published with explicit API surface for bitmap <-> tensor conversions across platforms.

### Changed
- BatchNorm now reshapes stats for broadcasting and exercises JVM/native tests; CPU backend implements `sqrt` to support it.

### Dependencies
- Added pbandk runtime 0.16.0 for ONNX protobuf decoding.

## [0.4.0] - 2025-12-03

### Added
- Recording/tracing pipeline for tensor ops (RecordingExecution/TracingTensorOps) and compute-graph DAG under `sk.ainet.lang.graph`, including tape-to-graph conversion and GraphViz export helpers/tests.
- JSON export proof of concept via new `skainet-compile-json` module with serialization models, `exportJson` CLI, and tiny graph golden fixtures.
- Multiplatform image IO module to convert platform bitmaps <-> tensors and RGB byte arrays; includes macOS implementation fixes.
- Dedicated YOLOv8 model module (`skainet-models:skainet-model-yolo`) with graph assembly, config/pre/post-processing, and missing upsample/concat ops required by the model.
- NN DSL additions: multi-input `Functional` wrapper, new `Upsample2d`/Softmax helpers, scalar DSL builder plus tensor/number operator overloads, and extra tensor view/pprint utilities.

### Changed
- Graph DSL relocated into the lang namespace with refreshed default execution tape/graph context wiring; removed unused integration module scaffolding.
- Removed committed MNIST training assets; rely on download at runtime.
- Added scalar arithmetic support across backends and void ops to match new operator overloads.

### Fixed
- Corrected unsqueeze view handling and data DSL dtype reuse; stabilized tracing/JSON/tape tests.
- Fixed macOS image conversion path and cleaned duplicate files in the new IO/image pipeline.

### Dependencies
- io.ktor client 3.3.3 (from 3.3.2).
- logback-classic 1.5.21 (from 1.5.20).

## [0.3.0] - 2025-11-27

### Added
- Kolmogorov–Arnold Network (KAN/AKN) module and DSL support, including public factory and aliases for direct construction. Introduces `Akn`/`AknConfig` and `createAkn` mirroring DSL defaults.
- Example KAN models and graphs (e.g., Sine function examples and pretrained variant) with tests and Graphviz export.
- Additional NN DSL conveniences around initialization scopes (weights/basis/bias) and activation hooks used by KAN.

### Changed
- Minor API refinements in lang/nn DSL to better align with execution context usage for new KAN modules.

### Fixed
- Stabilized integration tests for KAN modules and examples.

### Performance
- Minor initialization performance tweaks for new modules.

### Docs
- Updated docs and samples to include KAN usage and references.



## [0.2.0] - 2025-11-16

### Added
- Initial support for model code sharing API (model definition, execution, loading). Implements #196, related to #169.
- Batch Normalization layer. Implements #193.
- Forward hooks and simple tape recording for NN. Implements #190, related to #104.
- Common traversal base for modules, with tests; Embedding implementation with dual value types; switched EEmbeddings to DualModule implementation.
- Dropout (initial implementation) and phase support (training/eval) in execution context so modules can behave differently by phase. Related to #5.
- `tril` op (initial version).
- MaxPool op with DSL support; Conv2D DSL support.
- Data API: initial version including MNIST data loader; JSON loading support (renamed loader classes from CSV to JSON) with tests. Implements #180, #181; related to #176, #179.
- GGUF model loading implementation (initial import and working version). Implements #178, #182; related to #176, #177.
- MatMul support in backends.
- Nested data blocks support in DSL (data block returns a tensor); contexts for creating and collecting tensors (returning last or all created tensors).
- JVM Ops using the Java Vector API (initial implementation) and SIMD Vector API acceleration.
- JMH benchmarks (JVM module) and additional benchmarks.
- Sample showing general tensor calculations (e.g., image color transformations).

### Changed
- NN DSL refactored to use `ExecutionContext`; added `ExecutionContext` parameter to `forward` functions.
- Models and data APIs improved; unified tensor value creation in DSL; moved tensor creation context for safer vector/matrix/tensor creation.
- Default CPU compute used for JS target.
- JS and WASM Kotlin targets aligned for library packaging.
- Gradle updated to 9.0.0; Android target namespaces fixed.

### Fixed
- Crash in schema validation task; added Kotlin compiler plugin configuration for expect/actual.
- Activation not applied in Dense layer (fixed).
- JVM target issues; fixed failing JVM tests; added regression tests; stabilized platform matching test (temporarily ignored) and additional general test fixes.
- Miscellaneous build-signing validation added to avoid CI failures.

### Performance
- SIMD/Java Vector API acceleration for JVM backend operations.

### Dependencies
- com.vanniktech.maven.publish: 0.34.0 → 0.35.0.
- io.ktor (android, cio, content-negotiation, core, darwin, js, logging): 3.3.1 → 3.3.2.
- com.fasterxml.jackson.core:jackson-databind: 2.15.2 → 2.20.0 → 2.20.1.

### Build & CI
- GitHub Actions: use Java 22.
- Bump actions/checkout from v4 to v5.
- Add Gradle local caches to .gitignore.
- Preparations for 0.2.0 release and ability to build local Maven version of the upcoming release.

### Docs
- Added hint/reference on normalization layer paper. Related to #192.


## [0.1.0] - 2025-10-31
- Initial public release of SKaiNET 0.1.0.
