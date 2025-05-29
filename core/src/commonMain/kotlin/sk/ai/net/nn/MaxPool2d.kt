package sk.ai.net.nn

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.impl.DoublesTensor

/**
 * 2D max pooling layer.
 * Works with tensors of shape (N, C, H, W) or (C, H, W).
 */
class MaxPool2d(
    val kernelSize: Int,
    val stride: Int = kernelSize,
    override val name: String = "MaxPool2d"
) : Module() {
    override val modules: List<Module>
        get() = emptyList()

    override fun forward(input: Tensor): Tensor = maxPool2d(input)

    private fun maxPool2d(input: Tensor): Tensor {
        val tensor = input as DoublesTensor
        val shape = tensor.shape
        require(shape.rank == 3 || shape.rank == 4) {
            "MaxPool2d expected 3D or 4D input tensor, but got shape $shape"
        }
        val batchSize: Int
        val channels: Int
        val height: Int
        val width: Int
        if (shape.rank == 4) {
            batchSize = shape[0]
            channels = shape[1]
            height = shape[2]
            width = shape[3]
        } else {
            batchSize = 1
            channels = shape[0]
            height = shape[1]
            width = shape[2]
        }

        val outH = (height - kernelSize) / stride + 1
        val outW = (width - kernelSize) / stride + 1
        val outElements = DoubleArray(batchSize * channels * outH * outW)
        var idx = 0
        for (n in 0 until batchSize) {
            for (c in 0 until channels) {
                for (i in 0 until outH) {
                    for (j in 0 until outW) {
                        var maxVal = Double.NEGATIVE_INFINITY
                        for (ki in 0 until kernelSize) {
                            for (kj in 0 until kernelSize) {
                                val h = i * stride + ki
                                val w = j * stride + kj
                                val value = if (shape.rank == 4) {
                                    tensor[n, c, h, w]
                                } else {
                                    tensor[c, h, w]
                                }
                                if (value > maxVal) maxVal = value
                            }
                        }
                        outElements[idx++] = maxVal
                    }
                }
            }
        }
        val outShape = Shape(batchSize, channels, outH, outW)
        return DoublesTensor(outShape, outElements)
    }
}
