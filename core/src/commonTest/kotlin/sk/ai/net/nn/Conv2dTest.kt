package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.impl.DoublesTensor
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Conv2dTest {
    @Test
    fun testBasicConv2d() {
        // Create a simple 1x1x3x3 input tensor (1 batch, 1 channel, 3x3 spatial dimensions)
        val input = DoublesTensor(
            Shape(1, 1, 3, 3),
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            )
        )

        // Create a Conv2d layer with 1 input channel, 1 output channel, 2x2 kernel, no stride, no padding
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 1,
            kernelSize = 2,
            stride = 1,
            padding = 0,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x1x2x2 (1 batch, 1 channel, 2x2 spatial dimensions)
        assertEquals(Shape(1, 1, 2, 2), result.shape)

        // Since we can't control the weights, we can't check exact values
        // But we can check that the output has the right shape and contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dWithStride() {
        // Create a simple 1x1x4x4 input tensor
        val input = DoublesTensor(
            Shape(1, 1, 4, 4),
            doubleArrayOf(
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0
            )
        )

        // Create a Conv2d layer with stride=2
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 1,
            kernelSize = 2,
            stride = 2,
            padding = 0,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x1x2x2 (stride=2 reduces spatial dimensions by half)
        assertEquals(Shape(1, 1, 2, 2), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dWithPadding() {
        // Create a simple 1x1x3x3 input tensor
        val input = DoublesTensor(
            Shape(1, 1, 3, 3),
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            )
        )

        // Create a Conv2d layer with padding=1
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 1,
            kernelSize = 3,
            stride = 1,
            padding = 1,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x1x3x3 (padding preserves spatial dimensions)
        assertEquals(Shape(1, 1, 3, 3), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dMultipleChannels() {
        // Create a 1x2x3x3 input tensor (1 batch, 2 channels, 3x3 spatial dimensions)
        val input = DoublesTensor(
            Shape(1, 2, 3, 3),
            doubleArrayOf(
                // Channel 1
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                // Channel 2
                9.0, 8.0, 7.0,
                6.0, 5.0, 4.0,
                3.0, 2.0, 1.0
            )
        )

        // Create a Conv2d layer with 2 input channels, 1 output channel
        val conv = Conv2d(
            inChannels = 2,
            outChannels = 1,
            kernelSize = 2,
            stride = 1,
            padding = 0,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x1x2x2
        assertEquals(Shape(1, 1, 2, 2), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dMultipleOutputChannels() {
        // Create a 1x1x3x3 input tensor
        val input = DoublesTensor(
            Shape(1, 1, 3, 3),
            doubleArrayOf(
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            )
        )

        // Create a Conv2d layer with 1 input channel, 2 output channels
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 2,
            kernelSize = 2,
            stride = 1,
            padding = 0,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x2x2x2 (1 batch, 2 output channels, 2x2 spatial dimensions)
        assertEquals(Shape(1, 2, 2, 2), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dBatchProcessing() {
        // Create a 2x1x3x3 input tensor (2 batches, 1 channel, 3x3 spatial dimensions)
        val input = DoublesTensor(
            Shape(2, 1, 3, 3),
            doubleArrayOf(
                // Batch 1
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0,
                // Batch 2
                9.0, 8.0, 7.0,
                6.0, 5.0, 4.0,
                3.0, 2.0, 1.0
            )
        )

        // Create a Conv2d layer
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 1,
            kernelSize = 2,
            stride = 1,
            padding = 0,
            useBias = true
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 2x1x2x2 (2 batches, 1 channel, 2x2 spatial dimensions)
        assertEquals(Shape(2, 1, 2, 2), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }

    @Test
    fun testConv2dOutputSizeCalculation() {
        // Test various combinations of input size, kernel size, stride, and padding
        // to verify that the output size is calculated correctly

        // Case 1: No padding, stride 1
        val input1 = DoublesTensor(Shape(1, 1, 5, 5), DoubleArray(25) { 1.0 })
        val conv1 = Conv2d(inChannels = 1, outChannels = 1, kernelSize = 3, stride = 1, padding = 0)
        val result1 = conv1.forward(input1) as DoublesTensor
        assertEquals(Shape(1, 1, 3, 3), result1.shape) // (5 - 3) / 1 + 1 = 3

        // Case 2: With padding, stride 1
        val input2 = DoublesTensor(Shape(1, 1, 5, 5), DoubleArray(25) { 1.0 })
        val conv2 = Conv2d(inChannels = 1, outChannels = 1, kernelSize = 3, stride = 1, padding = 1)
        val result2 = conv2.forward(input2) as DoublesTensor
        assertEquals(Shape(1, 1, 5, 5), result2.shape) // (5 + 2*1 - 3) / 1 + 1 = 5

        // Case 3: No padding, stride 2
        val input3 = DoublesTensor(Shape(1, 1, 5, 5), DoubleArray(25) { 1.0 })
        val conv3 = Conv2d(inChannels = 1, outChannels = 1, kernelSize = 3, stride = 2, padding = 0)
        val result3 = conv3.forward(input3) as DoublesTensor
        assertEquals(Shape(1, 1, 2, 2), result3.shape) // (5 - 3) / 2 + 1 = 2

        // Case 4: With padding, stride 2
        val input4 = DoublesTensor(Shape(1, 1, 5, 5), DoubleArray(25) { 1.0 })
        val conv4 = Conv2d(inChannels = 1, outChannels = 1, kernelSize = 3, stride = 2, padding = 1)
        val result4 = conv4.forward(input4) as DoublesTensor
        assertEquals(Shape(1, 1, 3, 3), result4.shape) // (5 + 2*1 - 3) / 2 + 1 = 3
    }

    @Test
    fun testConv2dWithNoBias() {
        // Create a simple input tensor
        val input = DoublesTensor(Shape(1, 1, 3, 3), DoubleArray(9) { 1.0 })

        // Create a Conv2d layer with no bias
        val conv = Conv2d(
            inChannels = 1,
            outChannels = 1,
            kernelSize = 2,
            stride = 1,
            padding = 0,
            useBias = false
        )

        // Apply convolution
        val result = conv.forward(input) as DoublesTensor

        // Expected output shape: 1x1x2x2
        assertEquals(Shape(1, 1, 2, 2), result.shape)

        // Check that the output contains non-zero values
        assertTrue(result.elements.any { it != 0.0 }, "Output should contain non-zero values")
    }
}
