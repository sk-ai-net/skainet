package sk.ai.net.nn

import sk.ai.net.Shape
import sk.ai.net.Tensor
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

        // Apply padding if needed
        val paddedInput: Tensor = if (padding > 0) {
            val paddedH = inH + 2 * padding
            val paddedW = inW + 2 * padding
            val temp = zeros(Shape(batchSize, inC, paddedH, paddedW))
            for (n in 0 until batchSize) {
                for (c in 0 until inC) {
                    for (i in 0 until inH) {
                        for (j in 0 until inW) {
                            // temp[n, c, i + padding, j + padding] = input[n, c, i, j]
                        }
                    }
                }
            }
            temp
        } else {
            input  // no padding needed
        }

        // Prepare output tensor
        val output = zeros(Shape(batchSize, outChannels, outH, outW))

        // Convolution: iterate over batch, out channels, and output spatial positions
        for (n in 0 until batchSize) {
            for (oc in 0 until outChannels) {
                val biasVal = 0f // if (bias != null) 0f// bias[oc] else 0f
                for (i in 0 until outH) {
                    for (j in 0 until outW) {
                        var sum = 0f
                        // Sum over all input channels and kernel elements
                        for (c in 0 until inChannels) {
                            for (ki in 0 until kernelSize) {
                                for (kj in 0 until kernelSize) {
                                    //sum += paddedInput[n, c, i * stride + ki, j * stride + kj] *                                            weight[oc, c, ki, kj]
                                }
                            }
                        }
                        // Add bias and assign to output
                        //output[n, oc, i, j] = sum //+ biasVal
                    }
                }
            }
        }
        return output
    }
}
