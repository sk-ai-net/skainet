package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelRegistry
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The dispatcher must populate itself on first use, with no bootstrap call from the consumer.
 *
 * Before this, `KernelDispatch` was the only registry that did not self-heal: `KernelRegistry`
 * installs providers lazily via `DefaultCpuOpsJvm.ensureKernelProviders()`, but every dispatch
 * consumer had to remember `KernelPacks.install()` + `FfmRowMajorKernelPack.install()` before its
 * first forward pass. Forgetting it is invisible — the decoding reference kernel is correct, just
 * ~1000x slower — and it was forgotten by application entry points and diagnostic harnesses alike.
 */
class KernelDispatchSelfHealTest {

    @Test
    fun cold_dispatch_installs_providers_and_discovered_packs() {
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()

        // No bootstrap of any kind — exactly what a forgetful consumer does.
        KernelDispatch.ensureInstalled()

        val names = KernelDispatch.kernels().map { it.name }
        assertTrue(names.isNotEmpty(), "self-heal must register kernels")
        assertTrue(
            KernelRegistry.providers().isNotEmpty(),
            "providers must be discovered first, since KernelPacks.install() derives from them",
        )
        assertTrue(
            names.any { it.endsWith("-fp32") },
            "provider-derived dense FP32 view kernels expected; got $names",
        )
        assertTrue(
            names.any { it.startsWith("ffm-rowmajor-") },
            "ServiceLoader-discovered ViewKernelPack (FFM row-major) expected; got $names",
        )
        assertTrue(
            KernelDispatch.mappedServableEncodings().isNotEmpty(),
            "row-major pack should make mapped K-quant weights servable zero-copy",
        )
        println("SELFHEAL n=${names.size} providers=${KernelRegistry.availableNames()} kernels=${names.sorted()}")
    }

    @Test
    fun cold_dispatch_discovers_the_ternary_packs() {
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()

        // #1240's acceptance: the ternary packs must arrive through discovery alone — no
        // NativeTernaryF32GemvKernel.install() / NativeTernaryLmheadKernel.install() anywhere.
        KernelDispatch.ensureInstalled()

        // The packs register only when the bundled native library resolves; on a machine
        // without it, discovery must cost a lookup and register nothing (no crash, no entry).
        if (!NativeTernaryF32GemvKernel.isAvailable()) {
            println("SELFHEAL-TERNARY skipped: bundled native library unavailable")
            return
        }
        val names = KernelDispatch.kernels().map { it.name }
        assertTrue(
            names.any { it.startsWith("ternary_f32_gemv/") },
            "exact FP32×BITNET_B1_58 gemv expected after self-heal; got $names",
        )
        assertTrue(
            names.any { it.startsWith("ternary_planes_matmul/") },
            "fused BITNET_PLANES lm_head expected after self-heal; got $names",
        )
        assertTrue(
            KernelDispatch.find(sk.ainet.backend.api.kernel.TernaryF32GemvKernel.keyFor()) != null,
            "matmul(FP32 × BITNET_B1_58) must resolve to the LUT kernel with zero explicit installs",
        )
    }

    @Test
    fun explicit_registration_suppresses_auto_install() {
        KernelDispatch.clearForTesting()
        KernelRegistry.clearForTesting()
        // A consumer that wires its own kernels keeps full control: auto-install must not run
        // behind its back and silently add tiers it deliberately left out.
        sk.ainet.backend.api.kernel.KernelPacks.installReference()
        val afterExplicit = KernelDispatch.kernels().map { it.name }
        KernelDispatch.ensureInstalled()
        assertTrue(
            KernelDispatch.kernels().map { it.name } == afterExplicit,
            "ensureInstalled() must be a no-op once the table is non-empty",
        )
    }
}
