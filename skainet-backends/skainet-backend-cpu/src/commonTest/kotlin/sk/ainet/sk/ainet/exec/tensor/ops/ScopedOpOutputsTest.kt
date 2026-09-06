package sk.ainet.sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.forwardScope
import sk.ainet.lang.memory.StorageClosedException
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * #1146: op *outputs* draw from the active scope. Creation learned this in #1145; these tests pin
 * the other half — a `matmul`/`add`/`relu` chain on scope-bound tensors produces slab-backed
 * results, steady-state decode reuses the slab exactly (flat `peakFloats`, zero overflow), the
 * numbers are bit-identical to Ambient, and a stale read after `reset()` is loud.
 */
class ScopedOpOutputsTest {

    private val xVals = FloatArray(8) { (it - 3).toFloat() * 0.5f }         // [2, 4]
    private val wVals = FloatArray(12) { ((it * 7) % 5 - 2).toFloat() }     // [4, 3]
    private val bVals = floatArrayOf(0.1f, -0.2f, 0.3f)                     // [3]

    private fun <V> step(ctx: sk.ainet.context.ExecutionContext, x: Tensor<FP32, V>, w: Tensor<FP32, V>, b: Tensor<FP32, V>): Tensor<FP32, V> {
        val ops = x.ops
        return ops.relu(ops.add(ops.matmul(x, w), b))
    }

    private fun ambientResult(): FloatArray {
        val ctx = DirectCpuExecutionContext()
        val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
        val w = ctx.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class, wVals)
        val b = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, bVals)
        return step(ctx, x, w, b).data.copyToFloatArray()
    }

    @Test
    fun ambientOutputsAreUntouched() {
        val ctx = DirectCpuExecutionContext()
        val a = ctx.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
        val out = a.ops.add(a, a)
        assertFalse(out.data is StorageFloatTensorData<*>, "Ambient op outputs stay on the plain array path")
    }

    @Test
    fun opOutputsDrawFromTheSlabAndMatchAmbient() {
        val expected = ambientResult()
        DirectCpuExecutionContext().forwardScope(slabFloats = 256) { scoped, scope ->
            val x = scoped.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
            val w = scoped.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class, wVals)
            val b = scoped.fromFloatArray<FP32, Float>(Shape(3), FP32::class, bVals)
            val before = scope.usedFloats
            val y = step(scoped, x, w, b)
            assertTrue(y.data is StorageFloatTensorData<*>, "op output must be slab-backed under a scope")
            assertTrue(scope.usedFloats > before, "the chain must have drawn its outputs from the slab")
            assertContentEquals(expected, y.data.copyToFloatArray(), "slab-backed chain must match Ambient")
        }
    }

    @Test
    fun steadyStateDecodeIsAFlatLine() {
        val expected = ambientResult()
        val base = DirectCpuExecutionContext()
        // The real shape of a decode loop: weights persist OUTSIDE the forward scope
        // (Ambient here; ModelScope in a real model), activations live inside it.
        val w = base.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class, wVals)
        val b = base.fromFloatArray<FP32, Float>(Shape(3), FP32::class, bVals)
        base.forwardScope(slabFloats = 256) { scoped, scope ->
            var peakAfterWarmup = -1
            repeat(8) { stepNo ->
                val x = scoped.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
                val y = step(scoped, x, w, b)
                assertTrue(y.data is StorageFloatTensorData<*>, "step $stepNo: output must be slab-backed")
                assertContentEquals(expected, y.data.copyToFloatArray(), "step $stepNo numerics")
                if (stepNo == 1) peakAfterWarmup = scope.peakFloats
                if (stepNo > 1) {
                    assertEquals(peakAfterWarmup, scope.peakFloats, "step $stepNo: steady state must not grow the slab")
                }
                assertEquals(0L, scope.overflowBytes, "step $stepNo: nothing may spill past the slab")
                scope.reset()
            }
            assertEquals(8, scope.steps.toInt())
        }
    }

    @Test
    fun staleOpOutputThrowsAfterReset() {
        DirectCpuExecutionContext().forwardScope(slabFloats = 128) { scoped, scope ->
            val a = scoped.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
            val out = a.ops.add(a, a)
            scope.reset()
            assertFailsWith<StorageClosedException> { out.data[0, 0] }
        }
    }

    @Test
    fun slabOverflowStillComputesCorrectly() {
        val expected = ambientResult()
        DirectCpuExecutionContext().forwardScope(slabFloats = 4) { scoped, scope ->
            val x = scoped.fromFloatArray<FP32, Float>(Shape(2, 4), FP32::class, xVals)
            val w = scoped.fromFloatArray<FP32, Float>(Shape(4, 3), FP32::class, wVals)
            val b = scoped.fromFloatArray<FP32, Float>(Shape(3), FP32::class, bVals)
            val y = step(scoped, x, w, b)
            assertTrue(scope.overflowBytes > 0, "a 4-float slab must overflow")
            assertContentEquals(expected, y.data.copyToFloatArray(), "overflow path numerics")
        }
    }
}
