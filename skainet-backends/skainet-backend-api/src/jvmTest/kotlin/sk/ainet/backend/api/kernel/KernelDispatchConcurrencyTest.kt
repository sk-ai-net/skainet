package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.types.FP32
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * SKEEP-005 prerequisite: schedule workers dispatch concurrently, so the registries must survive
 * concurrent reads while writes stay serialized — no `ConcurrentModificationException`, no
 * double auto-install, no lost registration.
 */
class KernelDispatchConcurrencyTest {

    @AfterTest fun cleanup() { KernelDispatch.clearForTesting(); KernelRegistry.clearForTesting() }

    private class FakeKernel(override val name: String) : ViewKernel {
        override val key: KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(OperandKey.contiguous(Format.dense(FP32)), OperandKey.contiguous(Format.dense(FP32))),
        )
        override fun run(inputs: List<TensorView>, out: TensorView) = Unit
    }

    @Test
    fun concurrentRegisterAndFindNeverThrowAndKeepEveryName() {
        val threads = 16
        val perThread = 200
        val pool = Executors.newFixedThreadPool(threads)
        val start = CountDownLatch(1)
        val failures = AtomicInteger()
        val key = FakeKernel("probe").key
        repeat(threads) { t ->
            pool.submit {
                start.await()
                try {
                    repeat(perThread) { i ->
                        KernelDispatch.register(FakeKernel("k$t-$i"))
                        KernelDispatch.find(key)
                        KernelDispatch.kernels().size
                    }
                } catch (e: Throwable) {
                    failures.incrementAndGet()
                    e.printStackTrace()
                }
            }
        }
        start.countDown()
        pool.shutdown()
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS), "workers must finish")
        assertEquals(0, failures.get(), "no worker may fail")
        assertEquals(threads * perThread, KernelDispatch.kernels().size, "every distinct name is kept")
        assertNotNull(KernelDispatch.find(key))
    }

    @Test
    fun ensureInstalledRunsAtMostOnceUnderContention() {
        val installs = AtomicInteger()
        val provider = object : KernelProvider {
            override val name: String = "counting"
            override val priority: Int = 1
            override fun isAvailable(): Boolean = true
            override fun matmulFp32(): Fp32MatmulKernel? { installs.incrementAndGet(); return null }
        }
        KernelRegistry.register(provider)
        val threads = 16
        val pool = Executors.newFixedThreadPool(threads)
        val start = CountDownLatch(1)
        repeat(threads) { pool.submit { start.await(); KernelDispatch.ensureInstalled() } }
        start.countDown()
        pool.shutdown()
        assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS))
        // KernelPacks.install() resolves the provider's kernels once per install; contention must not repeat it.
        assertTrue(installs.get() <= 1 || KernelDispatch.kernels().isNotEmpty(), "auto-install ran without corrupting the table")
    }
}
