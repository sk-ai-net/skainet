package sk.ai.net.graph.tensor.shape


data class Shape(val dimensions: IntArray) {
    companion object {
        operator fun invoke(vararg dimensions: Int): Shape {
            return Shape(dimensions.copyOf())
        }
    }

    val volume: Int
        get() = dimensions.fold(1) { a, x -> a * x }

    val rank: Int
        get() = dimensions.size

    internal fun index(indices: IntArray): Int {
        assert(
            { indices.size == dimensions.size },
            { "`indices.size` must be ${dimensions.size}: ${indices.size}" })
        return dimensions.zip(indices).fold(0) { a, x ->
            assert(
                { 0 <= x.second && x.second < x.first },
                { "Illegal index: indices = ${indices}, shape = $dimensions" })
            a * x.first + x.second
        }
    }

    operator fun get(index: Int): Int {
        return dimensions[index]
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Shape) return false

        return dimensions.contentEquals(other.dimensions)
    }

    override fun hashCode(): Int {
        return dimensions.contentHashCode()
    }

    override fun toString(): String {
        // Create a string representation of the dimensions array
        val dimensionsString = dimensions.joinToString(separator = " x ", prefix = "[", postfix = "]")
        // Return the formatted string including dimensions and volume
        return "Shape: Dimensions = $dimensionsString, Size (Volume) = $volume"
    }
}

internal inline fun <R> zipFold(a: FloatArray, b: FloatArray, initial: R, operation: (R, Float, Float) -> R): R {
    var result: R = initial
    for (i in a.indices) {
        result = operation(result, a[i], b[i])
    }
    return result
}

internal inline fun <R> zipFold(a: IntArray, b: IntArray, initial: R, operation: (R, Int, Int) -> R): R {
    var result: R = initial
    for (i in a.indices) {
        result = operation(result, a[i], b[i])
    }
    return result
}

internal inline fun assert(value: () -> Boolean, lazyMessage: () -> Any) {
//    if (Tensor::class) {
    if (!value()) {
        val message = lazyMessage()
        throw AssertionError(message)
    }
    //  }
}
