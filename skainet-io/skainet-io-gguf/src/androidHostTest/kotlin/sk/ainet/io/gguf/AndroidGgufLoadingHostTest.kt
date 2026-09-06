package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.DeviceMemory
import sk.ainet.lang.memory.plan.PlannerProfile
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.MmapFloatTensorData
import sk.ainet.lang.types.FP32
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Host-side test of the *Android compilation* (#1038, SKEEP-002): this source set compiles against
 * androidMain, so it proves the Android loading facade builds and behaves on the Android variant —
 * mapped staging by default, and a fit check that answers before the load rather than during it.
 *
 * `AndroidGguf.deviceMemory(context)` needs a real `Context` and belongs to the instrumented smoke
 * test; everything downstream of it takes a [DeviceMemory] so it can be checked here.
 */
class AndroidGgufLoadingHostTest {

    private val mb = 1024L * 1024L

    /**
     * A one-tensor GGUF v3 file with a known F32 payload. Written here rather than reused from
     * jvmTest's `SyntheticGguf`, which this compilation cannot see.
     */
    private fun model(elements: Int = 4096): File {
        val file = File.createTempFile("android-gguf-", ".gguf")
        file.deleteOnExit()
        val head = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)
        head.putInt(0x46554747)              // "GGUF"
        head.putInt(3)                       // version
        head.putLong(1)                      // tensor count
        head.putLong(1)                      // kv count
        val key = "general.architecture".encodeToByteArray()
        head.putLong(key.size.toLong()); head.put(key)
        head.putInt(GGUFValueType.STRING.value)
        val value = "test".encodeToByteArray()
        head.putLong(value.size.toLong()); head.put(value)
        val name = "w_f32".encodeToByteArray()
        head.putLong(name.size.toLong()); head.put(name)
        head.putInt(1)                       // rank
        head.putLong(elements.toLong())
        head.putInt(GGMLQuantizationType.F32.value)
        head.putLong(0L)                     // data offset
        val padding = (32 - (head.position() % 32)) % 32
        repeat(padding) { head.put(0) }

        RandomAccessFile(file, "rw").use { raf ->
            raf.write(head.array(), 0, head.position())
            val payload = ByteBuffer.allocate(elements * 4).order(ByteOrder.LITTLE_ENDIAN)
            repeat(elements) { payload.putFloat(it * 0.5f) }
            raf.write(payload.array())
        }
        return file
    }

    private fun load(loader: StreamingGgufParametersLoader): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val out = LinkedHashMap<String, Tensor<FP32, Float>>()
        runBlocking { loader.load<FP32, Float>(ctx, FP32::class) { name, t -> out[name] = t } }
        return out
    }

    @Test
    fun `the android loader maps weights by default`() {
        val f = model()
        try {
            val mapped = load(AndroidGguf.loader(f.absolutePath))
            assertTrue(
                mapped.getValue("w_f32").data is MmapFloatTensorData<*>,
                "dense F32 must come from file-backed pages on Android, got ${mapped.getValue("w_f32").data::class.simpleName}",
            )
            // and the heap path is still reachable, producing the same numbers
            val onHeap = load(AndroidGguf.loader(f.absolutePath, weightForm = WeightForm()))
            assertTrue(onHeap.getValue("w_f32").data is FloatArrayTensorData<*>)
            assertContentEquals(
                onHeap.getValue("w_f32").data.copyToFloatArray(),
                mapped.getValue("w_f32").data.copyToFloatArray(),
                "staging must not change the numbers",
            )
            val values = mapped.getValue("w_f32").data.copyToFloatArray()
            assertEquals(0f, values[0]); assertEquals(0.5f, values[1]); assertEquals(2047.5f, values[4095])
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the fit check reads the plan from the header and answers before loading`() {
        val f = model()
        try {
            val plentiful = DeviceMemory(
                totalRamBytes = 4096 * mb, availableRamBytes = 2048 * mb,
                heapMaxBytes = 512 * mb, heapUsedBytes = 32 * mb, lowMemoryThresholdBytes = 180 * mb,
            )
            val fit = AndroidGguf.fits(plentiful, f.absolutePath, ctx = 128)
            assertTrue(fit.fits, fit.render())
            assertTrue(fit.weightsMapped, "the Android default is mapped weights")
            assertTrue(fit.plan.weightsBytes > 0, "the plan comes from the header")

            // a phone with almost nothing left says so, and says which pool ran out
            val squeezed = plentiful.copy(availableRamBytes = 190 * mb, heapMaxBytes = 8 * mb, heapUsedBytes = 7 * mb)
            val tight = AndroidGguf.fits(squeezed, f.absolutePath, ctx = 4096)
            assertFalse(tight.fits, tight.render())
            assertEquals("managed heap", tight.blockingPool)
            assertTrue(tight.suggestions.isNotEmpty(), "a failing fit must say what to do")
        } finally {
            f.delete()
        }
    }

    @Test
    fun `the android plan follows the mobile profile by default`() {
        val f = model()
        try {
            val phone = DeviceMemory(
                totalRamBytes = 2048 * mb, availableRamBytes = 1600 * mb,
                heapMaxBytes = 512 * mb, heapUsedBytes = 40 * mb, lowMemoryThresholdBytes = 180 * mb,
            )
            val profiled = AndroidGguf.profiledPlan(f.absolutePath, ctx = 512, device = phone)
            assertEquals(PlannerProfile.MOBILE_2GB, profiled.profile, "Android plans as a 2 GB phone (#1039)")
            assertEquals(256, profiled.plan.input.prefillChunk)
            assertTrue(profiled.render().contains("profile mobile-2gb"), profiled.render())
            profiled.requireFits(phone)

            // and it refuses, before allocating anything, when the device cannot take it
            val tiny = phone.copy(availableRamBytes = 780 * mb, heapMaxBytes = 8 * mb, heapUsedBytes = 7 * mb)
            val refusal = assertFailsWith<IllegalStateException> {
                AndroidGguf.profiledPlan(f.absolutePath, ctx = 8192, device = tiny).requireFits(tiny)
            }
            assertTrue(refusal.message!!.contains("mobile-2gb"), refusal.message!!)
        } finally {
            f.delete()
        }
    }

    @Test
    fun `an unmapped load is charged for its weights, a mapped one is not`() {
        val f = model()
        try {
            val device = DeviceMemory(
                totalRamBytes = 2048 * mb, availableRamBytes = 900 * mb,
                heapMaxBytes = 512 * mb, heapUsedBytes = 40 * mb, lowMemoryThresholdBytes = 180 * mb,
            )
            val mapped = AndroidGguf.fits(device, f.absolutePath, ctx = 512, weightsMapped = true)
            val heap = AndroidGguf.fits(device, f.absolutePath, ctx = 512, weightsMapped = false)
            assertEquals(
                mapped.plan.weightsBytes,
                heap.heap.neededBytes - mapped.heap.neededBytes,
                "the difference between the two is exactly the weights",
            )
        } finally {
            f.delete()
        }
    }
}
