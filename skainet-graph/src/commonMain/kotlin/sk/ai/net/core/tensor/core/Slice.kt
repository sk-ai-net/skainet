package sk.ai.net.core.tensor.core

import sk.ai.net.core.tensor.Tensor


data class Slice(val tensor: Tensor, val dimensionIndex: Int, val startIndex: Long, val endIndex: Long) {
    fun toRange() = startIndex..endIndex
}

fun Slice.start() = startIndex

fun Slice.end() = endIndex

fun Slice.all() = startIndex..tensor.shape.dimensions[dimensionIndex]

fun Slice.range(): LongRange = startIndex..endIndex
