package sk.ainet.exec.kernel

import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.lang.memory.plan.StorageCapabilities
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Emits `kernel-support.json` (introspected from the registered `KernelProvider`s) and
 * **gates drift in the scalar floor**: the all-platform baseline coverage is auto-derived
 * from `ScalarKernelProvider.supports(...)`, so adding/removing a scalar packed kernel
 * without updating the docs fails this test (runs under `java-tests` in CI).
 *
 * The JSON is rendered to the Antora page `reference/kernel-support-matrix.adoc` by the
 * build-logic `generateKernelMatrix` task — the kernel-side analogue of the
 * `operators.json` → `ops-status-matrix.adoc` pipeline.
 *
 * The SIMD/native tiers (Panama, native-FFM) are env-availability-gated (`isAvailable()`
 * probes the JDK incubator module / the loaded `.so`), so their *capability* is declared
 * here (the single place to edit when a provider gains a kernel) rather than probed.
 */
class KernelSupportMatrixTest {

    @AfterTest
    fun reset() = KernelDispatch.clearForTesting()

    private val formats = listOf("Float32", "BFloat16", "Q8_0", "Q4_0", "Q4_K", "Q6_K", "Q5_K", "Q5_1", "Q5_0")

    // platform key (display) -> the set of providers (by source-set) reaching it.
    private val platforms = listOf("JVM", "Android", "Native·linux", "Native·apple", "JS/WASM")

    private data class Tier(val name: String, val priority: Int, val platforms: Set<String>, val formats: Set<String>)

    private fun scalarFormats(): Set<String> =
        formats.filter { ScalarKernelProvider.supports("matmul", listOf("Float32", it)) }.toSet()

    // Source-set -> platforms. commonMain reaches all; backend-cpu jvmMain -> {JVM} only:
    // Panama Vector (jdk.incubator.vector) is a JDK-only incubator module — ART has no
    // Vector API, so PlatformCpuOpsFactory.android (skainet-backend-cpu/androidMain)
    // registers ONLY ServiceLoader-discovered providers + the scalar floor, never Panama.
    // backend-native-cpu jvmMain -> {JVM} (the native module declares only jvm()).
    // native-jni: skainet-backend-jni-cpu AAR — same C kernels via JNI, Android
    // only, discovered via ServiceLoader from PlatformCpuOpsFactory.android (#920).
    private fun tiers(): List<Tier> = listOf(
        Tier("scalar", 0, platforms.toSet(), scalarFormats()),
        Tier("panama-vector", 50, setOf("JVM"),
            setOf("Float32", "BFloat16", "Q8_0", "Q4_0", "Q4_K", "Q6_K", "Q5_K", "Q5_1", "Q5_0")),
        Tier("native-ffm", 100, setOf("JVM"),
            setOf("Float32", "BFloat16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q5_1", "Q5_0")),
        Tier("native-jni", 100, setOf("Android"),
            setOf("Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q5_1", "Q5_0")),
        // native-cinterop: NativeKnKernelProvider (nativeMain) — the same C
        // kernels statically embedded into the K/N klibs (#941/#942). Linux
        // since 0.39.x; Apple (iosArm64/iosSimulatorArm64/macosArm64) since
        // #959, with runtime FEAT_DotProd dispatch on Apple (#958: A13+/
        // M-series fast path, A12 scalar-int fallback). Registration is
        // manual via installNativeKernels() — no ServiceLoader on K/N.
        Tier("native-cinterop", 100, setOf("Native·linux", "Native·apple"),
            setOf("Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q5_1", "Q5_0")),
    )

    /**
     * Mapped serving (#1189): kernels that read the weight in canonical row-major GGUF file
     * order straight from off-heap bytes (mmap/direct buffer) — the `_rm` symbols behind
     * `JniBufferPackedMatmulKernel` on Android. K/N and remaining formats: #1192 follow-ups.
     *
     * The JVM row is *derived* (#1193), not hand-kept: `generate_and_gate_support_matrix`
     * installs `FfmRowMajorKernelPack` and reads `KernelDispatch.mappedServableEncodings()`,
     * which collects every registered [sk.ainet.backend.api.kernel.MappedCapableKernel]'s
     * `BLOCKED_ROW_MAJOR` operand. That same test asserts the derived set against
     * `StorageCapabilities.MAPPED_SERVABLE_DEFAULT` (`skainet-lang-core`), which cannot depend on
     * this module to derive it directly — so this is the guard that catches the two drifting.
     * The Android row stays hand-declared (nothing on a JVM test run can probe whether the JNI
     * `.so` loads on a device); `FfmRowMajorKernelPack` and `JniMappedKernelPack` register the
     * identical symbol list on purpose, so keep this row equal to the derived one by hand.
     */
    private fun mappedTiers(): List<Tier> = listOf(
        Tier("native-jni-direct", 100, setOf("Android"),
            setOf("Q4_K", "Q6_K", "Q5_K", "Q8_0", "Q4_0", "Q5_0", "Q5_1")),
        Tier("ffm-rowmajor", 100, setOf("JVM"), KernelDispatch.mappedServableEncodings().map { it.name }.toSet()),
    )

    private fun best(fmt: String, platform: String, tiers: List<Tier>): String? =
        tiers.filter { platform in it.platforms && fmt in it.formats }.maxByOrNull { it.priority }?.name

    private fun renderJson(tiers: List<Tier>): String {
        val version = System.getProperty("skainet.version", "dev")
        val sb = StringBuilder()
        sb.append("{\n")
        sb.append("  \"schema\": \"https://skainet.ai/schemas/kernel-support/v1\",\n")
        sb.append("  \"version\": \"").append(version).append("\",\n")
        sb.append("  \"op\": \"matmul\",\n")
        sb.append("  \"inputDtype\": \"Float32\",\n")
        sb.append("  \"platforms\": [").append(platforms.joinToString(", ") { "\"$it\"" }).append("],\n")
        sb.append("  \"formats\": [\n")
        formats.forEachIndexed { i, fmt ->
            val cells = platforms.mapNotNull { p -> best(fmt, p, tiers)?.let { "\"$p\": \"$it\"" } }
            sb.append("    {\"name\": \"").append(fmt).append("\", \"byPlatform\": {")
                .append(cells.joinToString(", ")).append("}}")
            sb.append(if (i == formats.lastIndex) "\n" else ",\n")
        }
        sb.append("  ],\n")
        val mapped = mappedTiers()
        val mappedFormats = formats.filter { fmt -> mapped.any { fmt in it.formats } }
        sb.append("  \"mapped\": [\n")
        mappedFormats.forEachIndexed { i, fmt ->
            val cells = platforms.mapNotNull { p -> best(fmt, p, mapped)?.let { "\"$p\": \"$it\"" } }
            sb.append("    {\"name\": \"").append(fmt).append("\", \"byPlatform\": {")
                .append(cells.joinToString(", ")).append("}}")
            sb.append(if (i == mappedFormats.lastIndex) "\n" else ",\n")
        }
        sb.append("  ]\n}\n")
        return sb.toString()
    }

    @Test
    fun generate_and_gate_support_matrix() {
        val tiers = tiers()

        // Drift gate on the scalar floor (the all-platform baseline): the documented set
        // below must equal what the scalar provider actually carries. Update both together.
        assertEquals(
            setOf("Float32", "BFloat16", "Q8_0", "Q4_0", "Q4_K", "Q6_K", "Q5_K", "Q5_1", "Q5_0"),
            scalarFormats(),
            "ScalarKernelProvider coverage changed — update the declared sets + run ./gradlew generateKernelMatrix",
        )

        // Sanity: every provider singleton is a KernelProvider (compile-time anchor).
        val providers: List<KernelProvider> = listOf(ScalarKernelProvider, PanamaVectorKernelProvider, NativeKernelProvider)
        assertEquals(3, providers.size)

        // Drift gate on mapped serving (#1193): install the JVM-reachable mapped-capable
        // kernels and assert what they actually serve against StorageCapabilities'
        // hand-kept mirror — the two lists can't independently drift without failing this test.
        // Dense F32 is excluded: StorageCapabilities maps it too, but as element-view serving,
        // not a matmul kernel, so it has no row in KernelDispatch.mappedServableEncodings().
        FfmRowMajorKernelPack.install()
        assertEquals(
            StorageCapabilities.MAPPED_SERVABLE_DEFAULT - TensorEncoding.Dense(4),
            KernelDispatch.mappedServableEncodings(),
            "KernelDispatch.mappedServableEncodings() drifted from " +
                "StorageCapabilities.MAPPED_SERVABLE_DEFAULT — update both, and mappedTiers()'s " +
                "native-jni-direct row",
        )

        val jsonText = renderJson(tiers)
        val outDir = File("build/generated/kernel-support").apply { mkdirs() }
        File(outDir, "kernel-support.json").writeText(jsonText)
        println("KERNEL_SUPPORT_JSON_BEGIN\n$jsonText\nKERNEL_SUPPORT_JSON_END")
    }
}
