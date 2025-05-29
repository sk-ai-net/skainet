package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor
import sk.ai.net.rand
import sk.ai.net.zeros
import kotlin.math.sqrt

class Conv2d(
    val inChannels: Int,
    val outChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0,
    useBias: Boolean = true,
    name: String = "Conv2d"
) : Module() {
    override val name: String = name
    val weight: Tensor
    val bias: Tensor?
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor = con2d(input)

    init {
        // Initialize weights and bias
        val fanIn = inChannels * kernelSize * kernelSize
        val bound = 1f / sqrt(fanIn.toDouble()).toFloat()  // 1/sqrt(fanIn)
        // Weight: uniform in [-bound, bound]
        weight = (((rand(
            Shape(
                outChannels,
                inChannels,
                kernelSize,
                kernelSize
            )
        ) as DoublesTensor) * (2f * bound).toDouble()) as DoublesTensor) - bound.toDouble()
        // Bias: uniform in [-bound, bound] if enabled
        bias = if (useBias) {
            ((rand(Shape(outChannels)) as DoublesTensor) * (2f * bound).toDouble()) - bound.toDouble()
        } else {
            null
        }
    }

    fun con2d(input: Tensor): Tensor {
        // Ensure input has 3D or 4D shape
        val shape = input.shape  // assume shape is a list or array of dimensions
        require(shape.rank == 3 || shape.rank == 4) {
            "Conv2d expected 3D or 4D input tensor, but got shape ${shape}."
        }
        // Determine batch size and input dims
        val batchSize: Int
        val inC: Int
        val inH: Int
        val inW: Int
        if (shape.rank == 4) {
            batchSize = shape.dimensions[0]
            inC = shape[1]
            inH = shape[2]
            inW = shape[3]
        } else {
            // if 3D (C, H, W), treat as batch of size 1
            batchSize = 1
            inC = shape[0]
            inH = shape[1]
            inW = shape[2]
        }
        require(inC == inChannels) {
            "Conv2d expected input channel count $inChannels, but got $inC."
        }

        // Compute output spatial dimensions
        val outH = (inH + 2 * padding - kernelSize) / stride + 1
        val outW = (inW + 2 * padding - kernelSize) / stride + 1
        require(outH > 0 && outW > 0) {
            "Conv2d output size is invalid (outH=$outH, outW=$outW). Check input dimensions and padding."
        }

        // Since we can't directly set values in tensors, we'll use a different approach
        // We'll create a DoubleArray to hold the output values and then create a new tensor from it

        val outputElements = DoubleArray(batchSize * outChannels * outH * outW)

        // Convolution: iterate over batch, out channels, and output spatial positions
        for (n in 0 until batchSize) {
            for (oc in 0 until outChannels) {
                val biasVal = if (bias != null) (bias as DoublesTensor)[oc] else 0.0
                for (i in 0 until outH) {
                    for (j in 0 until outW) {
                        var sum = 0.0
                        // Sum over all input channels and kernel elements
                        for (c in 0 until inChannels) {
                            for (ki in 0 until kernelSize) {
                                for (kj in 0 until kernelSize) {
                                    // Calculate input position with padding consideration
                                    val inPosH = i * stride + ki - padding
                                    val inPosW = j * stride + kj - padding

                                    // Skip if outside input bounds (zero padding)
                                    if (inPosH >= 0 && inPosH < inH && inPosW >= 0 && inPosW < inW) {
                                        sum += (input as DoublesTensor)[n, c, inPosH, inPosW] *
                                                (weight as DoublesTensor)[oc, c, ki, kj]
                                    }
                                }
                            }
                        }

                        // Add bias and store in output array
                        val outputIndex = ((n * outChannels + oc) * outH + i) * outW + j
                        outputElements[outputIndex] = sum + biasVal
                    }
                }
            }
        }

        // Create and return the output tensor
        return DoublesTensor(Shape(batchSize, outChannels, outH, outW), outputElements)
    }
}
