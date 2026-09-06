package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** SKEEP-003 Phase 0: `AllocationSpec` replaced the never-consumed `StorageSpec` (deleted in #1142). */
class AllocationSpecTest {

    @Test
    fun bytesFollowTheEncoding() {
        assertEquals(4L * 1000, AllocationSpec(Format.dense(FP32), 1000).bytes)
        assertEquals(2L * 1000, AllocationSpec(Format.dense(BF16), 1000).bytes)
        assertEquals(1000L, AllocationSpec(Format.dense(Int8), 1000).bytes)
        // Q4_K: 144 bytes per 256 elements
        assertEquals(144L * 4, AllocationSpec(Format(FP32, TensorEncoding.Q4_K), 1024).bytes)
        assertEquals(144L, AllocationSpec(Format(FP32, TensorEncoding.Q4_K), 1).bytes) // partial block rounds up
        // Q8_0: 34 bytes per 32 elements
        assertEquals(34L * 2, AllocationSpec.of(Format(FP32, TensorEncoding.Q8_0), Shape(2, 32)).bytes)
        // TurboQuant 4-bit, block 128: seed(4) + 4 groups × 2 B scales + 64 B codes = 76 per block
        val tq = TensorEncoding.TurboQuantPolar(bitsPerElement = 4, blockSize = 128)
        assertEquals(tq.physicalBytes(256), AllocationSpec(Format(FP32, tq), 256).bytes)
    }

    @Test
    fun opaqueEncodingHasNoComputableSize() {
        val spec = AllocationSpec(Format(FP32, TensorEncoding.Opaque("IQ2_XXS", 0)), 64)
        // Opaque carries its raw byte count; zero is "unknown" → physicalBytes may be null or 0 depending on the encoding
        val b = spec.bytesOrNull
        assertTrue(b == null || b == 0L)
    }

    @Test
    fun defaultsAreAmbientHeapMutableAligned64() {
        val s = AllocationSpec(Format.dense(FP32), 8)
        assertEquals(MemoryDomain.HOST_HEAP, s.domain)
        assertEquals(ScopeKind.AMBIENT, s.scope)
        assertTrue(s.mutable)
        assertEquals(64, s.alignment)
        assertFalse(s.format.isDense.not())
    }

    @Test
    fun validation() {
        assertFailsWith<IllegalArgumentException> { AllocationSpec(Format.dense(FP32), -1) }
        assertFailsWith<IllegalArgumentException> { AllocationSpec(Format.dense(FP32), 1, alignment = 48) }
        assertFailsWith<IllegalArgumentException> { AllocationSpec(Format.dense(FP32), 1, alignment = 0) }
    }

    @Test
    fun weightSpecCanBeExpressedDirectly() {
        val weights = AllocationSpec(
            Format(FP32, TensorEncoding.Q4_K), 1024,
            domain = MemoryDomain.MMAP_FILE, scope = ScopeKind.MODEL, mutable = false
        )
        assertEquals(Format(FP32, TensorEncoding.Q4_K), weights.format)
        assertEquals(1024L, weights.elementCount)
        assertEquals(MemoryDomain.MMAP_FILE, weights.domain)
        assertEquals(ScopeKind.MODEL, weights.scope)
        assertFalse(weights.mutable)

        val owned = AllocationSpec(Format.dense(BF16), 10)
        assertEquals(ScopeKind.AMBIENT, owned.scope)
        assertTrue(owned.mutable)
        assertEquals(20L, owned.bytes)
    }

    @Test
    fun scopeKindHasTheThreeLifetimes() {
        assertEquals(listOf(ScopeKind.MODEL, ScopeKind.FORWARD, ScopeKind.AMBIENT), ScopeKind.entries)
        assertNull(ScopeKind.entries.firstOrNull { it.name == "DEVICE" })
    }
}
