package sk.ai.net.core

import sk.ai.net.graph.tensor.Tensor


data class Slice(val tensor: Tensor, val dimensionIndex: Int, val startIndex: Long, val endIndex: Long) {
    fun toRange() = startIndex..endIndex
}

fun Slice.start() = startIndex

fun Slice.end() = endIndex

fun Slice.all() = startIndex..tensor.shape.dimensions[dimensionIndex]

fun Slice.range(): LongRange = startIndex..endIndex
