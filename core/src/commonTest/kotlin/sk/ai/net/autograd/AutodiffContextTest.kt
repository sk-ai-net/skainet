package sk.ai.net.autograd

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class AutodiffContextTest {

    @Test
    fun testCreateTensorInTrainingMode() {
        AutodiffContext.withContext(AutodiffMode.TRAINING) { ctx ->
            // Create a tensor in training mode with requiresGrad = true
            val x = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)

            // Check that the tensor is an AutogradTensor and requires gradients
            assertTrue(x is AutogradTensor)
            assertTrue((x as AutogradTensor).requiresGrad)

            // Create a tensor in training mode with requiresGrad = false
            val y = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = false)

            // Check that the tensor is an AutogradTensor but doesn't require gradients
            assertTrue(y is AutogradTensor)
            assertFalse((y as AutogradTensor).requiresGrad)
        }
    }

    @Test
    fun testCreateTensorInInferenceMode() {
        AutodiffContext.withContext(AutodiffMode.INFERENCE) { ctx ->
            // Create a tensor in inference mode
            val x = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)

            // Check that the tensor is a DoublesTensor
            assertTrue(x is DoublesTensor)

            // Create another tensor in inference mode
            val y = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = false)

            // Check that the tensor is a DoublesTensor
            assertTrue(y is DoublesTensor)

            // Check that operations work correctly in inference mode
            val z = x + y
            assertTrue(z is DoublesTensor)

            // Check that the result is correct
            val expected = DoublesTensor(Shape(2, 2), doubleArrayOf(2.0, 4.0, 6.0, 8.0))
            assertEquals(expected, z)
        }
    }

    @Test
    fun testCreateTensorInssInferenceMode() {
        AutodiffContext.withContext(AutodiffMode.TRAINING) { ctx ->
            // Create a tensor in inference mode
            val x = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)

            // Create another tensor in inference mode
            val y = ctx.tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = false)

            val z = x * y

            val d = ctx.backprop(z)

        }
    }
}
