package sk.ainet.sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.forwardScope
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * #1145 guard for the offset-0 trap: a slab-backed tensor has a nonzero `arrayOffset`, and a
 * fast path that grabbed the raw buffer would read the whole slab. `StorageFloatTensorData`
 * deliberately stays off `FloatArrayTensorData`, so real CPU ops must produce numbers identical
 * to the Ambient path — for tensors sliced from anywhere in the slab, across resets.
 */
class ScopedCreationOpsParityTest {

    @Test
    fun opsOnSlabBackedTensorsMatchAmbient() {
        val ctx = DirectCpuExecutionContext()
        val aVals = FloatArray(6) { (it + 1).toFloat() }        // 2×3
        val bVals = FloatArray(12) { (it % 5 - 2).toFloat() }   // 3×4

        val ambientMatmul: FloatArray
        val ambientAdd: FloatArray
        run {
            val a = ctx.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, aVals)
            val b = ctx.fromFloatArray<FP32, Float>(Shape(3, 4), FP32::class, bVals)
            ambientMatmul = ctx.ops.matmul(a, b).data.copyToFloatArray()
            ambientAdd = ctx.ops.add(a, a).data.copyToFloatArray()
        }

        ctx.forwardScope(slabFloats = 64) { scoped, scope ->
            repeat(3) { step ->
                // A leading allocation pushes the later tensors deeper into the slab, so the
                // offsets under test are nonzero and different from the previous step's layout.
                scoped.zeros<FP32, Float>(Shape(1 + step), FP32::class)
                val a = scoped.fromFloatArray<FP32, Float>(Shape(2, 3), FP32::class, aVals)
                val b = scoped.fromFloatArray<FP32, Float>(Shape(3, 4), FP32::class, bVals)
                assertTrue(a.data is StorageFloatTensorData<*>, "step $step: creation must draw from the slab")
                assertTrue((a.data as StorageFloatTensorData<*>).storage.arrayOffset > 0, "offset under test must be nonzero")

                assertContentEquals(ambientMatmul, ctx.ops.matmul(a, b).data.copyToFloatArray(), "step $step: matmul")
                assertContentEquals(ambientAdd, ctx.ops.add(a, a).data.copyToFloatArray(), "step $step: add")
                assertEquals(Shape(2, 4), ctx.ops.matmul(a, b).shape)
                scope.reset()
            }
        }
    }
}
