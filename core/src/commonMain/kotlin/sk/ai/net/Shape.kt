package sk.ai.net

import sk.ai.net.impl.assert
import sk.ai.net.impl.zipFold

class Shape(vararg dimensions: Int) {
    val dimensions: IntArray = dimensions.copyOf()

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
        if (other !is Shape) {
            return false
        }

        return dimensions.size == other.dimensions.size && zipFold(dimensions, other.dimensions, true) { result, a, b ->
            if (!result) {
                return false
            }
            a == b
        }
    }

    override fun hashCode(): Int {
        return dimensions.hashCode()
    }

    override fun toString(): String {
        // Create a string representation of the dimensions array
        val dimensionsString = dimensions.joinToString(separator = " x ", prefix = "[", postfix = "]")
        // Return the formatted string including dimensions and volume
        return "Shape: Dimensions = $dimensionsString, Size (Volume) = $volume"
    }

}
