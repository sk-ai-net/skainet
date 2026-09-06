package sk.ainet.exec.tensor.ops

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.forwardScope
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

/**
 * #1173: the JVM Panama vector paths accept slab-backed operands at nonzero offsets instead of
 * silently falling back to the scalar loops. Every vector-eligible shape is driven with both
 * operands sliced from a `ForwardScope` slab (leading pad ⇒ nonzero `arrayOffset`) and compared
 * bit-for-bit against the Ambient result — the tightest possible guard against off-by-offset
 * reads, which produce plausible garbage rather than crashes.
 */
class SlabOperandVectorPathTest {

    private val n = 67 // deliberately not a multiple of any vector species length
    private val aVals = FloatArray(n) { (it % 13 - 6) * 0.25f }
    private val bVals = FloatArray(n) { (it % 7 - 3) * 0.5f }
    private val biasVals = FloatArray(n) { (it % 5 - 2) * 0.125f }

    private fun ambient(op: (ctx: DirectCpuExecutionContext) -> Tensor<FP32, Float>): FloatArray =
        op(DirectCpuExecutionContext()).data.copyToFloatArray()

    /** Runs [op] with all operands created inside a scope, after a pad allocation forcing nonzero offsets. */
    private fun slab(op: (ctx: sk.ainet.context.ExecutionContext) -> Tensor<FP32, Float>): FloatArray {
        lateinit var result: FloatArray
        DirectCpuExecutionContext().forwardScope(slabFloats = 4096) { scoped, _ ->
            scoped.zeros<FP32, Float>(Shape(5), FP32::class) // pad: everything after this has offset > 0
            val out = op(scoped)
            assertTrue(
                (out.data is StorageFloatTensorData<*>),
                "output should be slab-backed — otherwise this test is not testing the vector paths",
            )
            result = out.data.copyToFloatArray()
        }
        return result
    }

    private fun t(ctx: sk.ainet.context.ExecutionContext, shape: Shape, v: FloatArray): Tensor<FP32, Float> {
        val slice = FloatArray(shape.volume) { v[it] }
        return ctx.fromFloatArray(shape, FP32::class, slice)
    }

    private fun check(name: String, op: (ctx: sk.ainet.context.ExecutionContext) -> Tensor<FP32, Float>) {
        val expected = ambient { op(it) }
        val actual = slab(op)
        assertContentEquals(expected, actual, name)
    }

    @Test
    fun exactShapeBinary() = check("add exact") { ctx ->
        val a = t(ctx, Shape(n), aVals)
        val b = t(ctx, Shape(n), bVals)
        a.ops.add(a, b)
    }

    @Test
    fun scalarBroadcastBothSides() {
        check("a scalar") { ctx ->
            val a = t(ctx, Shape(1), floatArrayOf(1.5f))
            val b = t(ctx, Shape(n), bVals)
            a.ops.multiply(a, b)
        }
        check("b scalar") { ctx ->
            val a = t(ctx, Shape(n), aVals)
            val b = t(ctx, Shape(1), floatArrayOf(-0.75f))
            a.ops.subtract(a, b)
        }
    }

    @Test
    fun biasBroadcastBothSides() {
        check("bias on b") { ctx ->
            val a = t(ctx, Shape(3, n), FloatArray(3 * n) { aVals[it % n] })
            val b = t(ctx, Shape(n), biasVals)
            a.ops.add(a, b)
        }
        check("bias on a") { ctx ->
            val a = t(ctx, Shape(n), biasVals)
            val b = t(ctx, Shape(3, n), FloatArray(3 * n) { bVals[it % n] })
            a.ops.add(a, b)
        }
    }

    @Test
    fun unaryActivations() {
        check("relu") { ctx -> t(ctx, Shape(n), aVals).let { it.ops.relu(it) } }
        check("silu") { ctx -> t(ctx, Shape(n), aVals).let { it.ops.silu(it) } }
    }

    @Test
    fun reduceAllMatchesAmbientBitForBit() {
        // sum(null) returns a rank-0 scalar; both paths must accumulate in the same order.
        val expected = ambient { ctx -> t(ctx, Shape(n), aVals).let { it.ops.sum(it, null) } }
        lateinit var actual: FloatArray
        DirectCpuExecutionContext().forwardScope(slabFloats = 4096) { scoped, _ ->
            scoped.zeros<FP32, Float>(Shape(5), FP32::class)
            val x = t(scoped, Shape(n), aVals)
            actual = x.ops.sum(x, null).data.copyToFloatArray()
        }
        assertContentEquals(expected, actual, "reduce-all sum")
    }

    @Test
    fun matmul2dThroughTheKernelSpi() = check("matmul") { ctx ->
        val a = t(ctx, Shape(7, 9), FloatArray(63) { aVals[it % n] })
        val b = t(ctx, Shape(9, 5), FloatArray(45) { bVals[it % n] })
        a.ops.matmul(a, b)
    }
}
