package sk.ai.net.graph.tensor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.random.Random

class TernaryTensorTest {

    @Test
    fun testTernaryTensorCreationAndAccess() {
        // Create a simple 2x3 tensor with values
        val shape = sk.ai.net.graph.tensor.shape.Shape(2, 3)
        val values = doubleArrayOf(
            -1.0, 0.0, 1.0,
            0.5, -0.7, 0.2
        )

        // Create a SimpleTensor (using DoubleArray) for comparison
        val simpleTensor = SimpleTensor(shape, values)

        // Create a TernaryTensor from the same values
        val ternaryTensor = TernaryTensor.fromDoubles(shape, values)

        // Verify the shape is preserved
        assertEquals(shape, ternaryTensor.shape)

        // Verify values are correctly quantized to -1, 0, 1
        // Note: TernaryTensor quantizes values: >0.5 to 1, <-0.5 to -1, else to 0
        assertEquals(-1.0, ternaryTensor[0, 0])
        assertEquals(0.0, ternaryTensor[0, 1])
        assertEquals(1.0, ternaryTensor[0, 2])
        assertEquals(0.0, ternaryTensor[1, 0]) // 0.5 is quantized to 0
        assertEquals(-1.0, ternaryTensor[1, 1]) // -0.7 is quantized to -1
        assertEquals(0.0, ternaryTensor[1, 2]) // 0.2 is quantized to 0
    }

    @Test
    fun testTernaryTensorWithLargeRandomData() {
        // Create a larger tensor with random values
        val shape = sk.ai.net.graph.tensor.shape.Shape(10, 10, 10) // 1000 elements
        val values = DoubleArray(1000) { Random.nextDouble(-1.0, 1.0) }

        // Create tensors
        val simpleTensor = SimpleTensor(shape, values)
        val ternaryTensor = TernaryTensor.fromDoubles(shape, values)

        // Verify random access to elements
        for (i in 0 until 50) {
            val x = Random.nextInt(10)
            val y = Random.nextInt(10)
            val z = Random.nextInt(10)

            // Get the original value
            val originalValue = simpleTensor[x, y, z]

            // Get the ternary value
            val ternaryValue = ternaryTensor[x, y, z]

            // Verify the ternary value is correctly quantized
            val expectedValue = when {
                originalValue > 0.5 -> 1.0
                originalValue < -0.5 -> -1.0
                else -> 0.0
            }

            assertEquals(expectedValue, ternaryValue)
        }
    }

    @Test
    fun testMemoryUsage() {
        // Create tensors of different sizes
        val sizes = listOf(100, 1000, 10000, 100000)

        for (size in sizes) {
            val shape = sk.ai.net.graph.tensor.shape.Shape(size)
            val values = DoubleArray(size) { Random.nextDouble(-1.0, 1.0) }

            // Create tensors
            val simpleTensor = SimpleTensor(shape, values)
            val ternaryTensor = TernaryTensor.fromDoubles(shape, values)

            // Calculate memory usage
            val simpleMemoryUsage = size * 8 // 8 bytes per double
            val ternaryMemoryUsage = ternaryTensor.memoryUsage()

            // Print memory usage
            println("Size: $size")
            println("  SimpleTensor memory usage: $simpleMemoryUsage bytes")
            println("  TernaryTensor memory usage: $ternaryMemoryUsage bytes")
            println("  Memory reduction: ${(1 - ternaryMemoryUsage.toDouble() / simpleMemoryUsage) * 100}%")

            // Verify TernaryTensor uses significantly less memory
            assertTrue(ternaryMemoryUsage < simpleMemoryUsage / 3, 
                "TernaryTensor should use at least 3x less memory than SimpleTensor")
        }
    }
}
