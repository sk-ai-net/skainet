package sk.ainet.context

import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.StorageClosedException
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1145: `ExecutionContext.memoryScope` gets its first reader. Ambient is byte-for-byte the old
 * path; a ForwardScope makes creation draw from the slab, `reset()` recycles it, and a
 * use-after-reset is a loud [StorageClosedException] rather than silent garbage.
 */
class ExecutionContextScopeTest {

    @Test
    fun ambientContextIsUntouched() {
        val ctx = DefaultDataExecutionContext()
        val t = ctx.zeros<FP32, Float>(Shape(4), FP32::class)
        assertFalse(t.data is StorageFloatTensorData<*>, "Ambient must use the factory path")
    }

    @Test
    fun scopedCreationDrawsFromTheSlab() {
        DefaultDataExecutionContext().forwardScope(slabFloats = 64) { ctx, scope ->
            val t = ctx.zeros<FP32, Float>(Shape(2, 8), FP32::class)
            assertTrue(t.data is StorageFloatTensorData<*>, "scoped creation must use the slab")
            assertEquals(16, scope.usedFloats)
            val u = ctx.full<FP32, Float>(Shape(8), FP32::class, 3)
            assertEquals(24, scope.usedFloats, "second tensor bumps the same slab")
            assertEquals(0f, t.data[0, 0])
            assertEquals(3f, u.data[3])
        }
    }

    @Test
    fun theSlabIsDirtyAndZerosMustZero() {
        DefaultDataExecutionContext().forwardScope(slabFloats = 16) { ctx, scope ->
            val dirty = ctx.zeros<FP32, Float>(Shape(16), FP32::class)
            for (i in 0 until 16) dirty.data.set(i, value = 7f)
            scope.reset()
            val clean = ctx.zeros<FP32, Float>(Shape(16), FP32::class)
            for (i in 0 until 16) assertEquals(0f, clean.data[i], "slab byte $i must be re-zeroed")
        }
    }

    @Test
    fun fromFloatArrayCopiesIntoTheSlab() {
        DefaultDataExecutionContext().forwardScope(slabFloats = 8) { ctx, _ ->
            val t = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(1f, 2f, 3f, 4f))
            assertTrue(t.data is StorageFloatTensorData<*>)
            assertEquals(2f, t.data[1])
            assertEquals(4f, t.data[3])
        }
    }

    @Test
    fun steadyStateReusesTheSlabAndStaleReadsThrow() {
        DefaultDataExecutionContext().forwardScope(slabFloats = 32) { ctx, scope ->
            val step1 = ctx.ones<FP32, Float>(Shape(32), FP32::class)
            assertEquals(32, scope.peakFloats)
            scope.reset()
            assertEquals(0, scope.usedFloats, "reset rewinds the slab")
            val step2 = ctx.ones<FP32, Float>(Shape(32), FP32::class)
            assertEquals(32, scope.peakFloats, "steady state allocates zero new slab bytes")
            assertEquals(1f, step2.data[0])
            assertFailsWith<StorageClosedException>("a step-1 tensor read after reset must be loud") {
                step1.data[0]
            }
        }
    }

    @Test
    fun retainIsTheSanctionedEscape() {
        val ctx = DefaultDataExecutionContext()
        lateinit var kept: FloatArray
        ctx.forwardScope(slabFloats = 8) { scoped, scope ->
            val t = scoped.fromFloatArray<FP32, Float>(Shape(4), FP32::class, floatArrayOf(9f, 8f, 7f, 6f))
            val data = t.data as StorageFloatTensorData<*>
            val retained = scope.retain(data.storage)
            scope.reset()
            kept = retained.floats!!.copyOfRange(retained.arrayOffset, retained.arrayOffset + 4)
        }
        assertEquals(listOf(9f, 8f, 7f, 6f), kept.toList())
    }

    @Test
    fun overflowBeyondTheSlabStillWorks() {
        DefaultDataExecutionContext().forwardScope(slabFloats = 4) { ctx, scope ->
            val big = ctx.zeros<FP32, Float>(Shape(64), FP32::class)
            assertTrue(big.data is StorageFloatTensorData<*>)
            assertTrue(scope.overflowBytes > 0, "a tensor over the slab size goes to tracked overflow")
            assertEquals(0f, big.data[63])
        }
    }
}
