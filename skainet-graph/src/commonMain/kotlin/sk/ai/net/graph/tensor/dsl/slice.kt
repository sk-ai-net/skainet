package sk.ai.net.graph.tensor.dsl

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.graph.tensor.core.Slice
import sk.ai.net.core.TypedTensor
import kotlin.math.abs


/**
slices {
// from second to the last
slice {
from 2 to last
}
// all elemnts, equals to ":" in pandas
slice {
all
}
// from 0 to the second last element
slice {
to -2
}
// from 0 to the second last element
slice {
from 2  to -2
}
}

slices {
start {
all()
}
end {
}
}

 */

class SliceBuilder(private val tensor: Tensor, private val dimensionIndex: Int) {
    var startIndex: Long = 0
    var endIndex: Long = tensor.shape.dimensions[dimensionIndex].toLong()

    infix fun from(start: Int): FromBuilder {
        this.startIndex = start.toLong()
        return FromBuilder(this)
    }

    infix fun up(end: Int): FromBuilder {
        this.startIndex = 0
        if (end > 0) {
            this.endIndex = end.toLong()
        } else {
            this.endIndex = tensor.shape.dimensions[dimensionIndex] - abs(end.toLong())
        }
        return FromBuilder(this)
    }

    fun all() {
        this.startIndex = 0
        this.endIndex = tensor.shape.dimensions[dimensionIndex].toLong()
    }

    fun first() {
        this.startIndex = 0
        this.endIndex = 0
    }


    fun last() {
        this.startIndex = tensor.shape.dimensions[dimensionIndex].toLong()
        this.endIndex = tensor.shape.dimensions[dimensionIndex].toLong()
    }

    fun none() {
        this.startIndex = -1
        this.endIndex = 0
    }


    fun build() = Slice(tensor, dimensionIndex, startIndex, endIndex)

    inner class FromBuilder(private val sliceBuilder: SliceBuilder) {
        infix fun to(end: Int) {
            sliceBuilder.endIndex = if (end == -1) tensor.shape.dimensions[dimensionIndex].toLong() else end.toLong()
        }
    }
}

class SlicesBuilder(private val tensor: Tensor) {
    private val slices = mutableListOf<Slice>()

    fun segment(init: SliceBuilder.() -> Unit) {
        val dimensionIndex = slices.size
        val builder = SliceBuilder(tensor, dimensionIndex)
        builder.init()
        slices.add(builder.build())
    }

    fun build() = slices
}

fun <T> slice(tensor: TypedTensor<T>, init: SlicesBuilder.() -> Unit): TypedTensor<T> {
    val builder = SlicesBuilder(tensor)
    builder.init()
    val slices = builder.build()
    return tensor.get(*slices.toTypedArray())
}

fun printVarargs(vararg elements: Slice) {
    for (element in elements) {
        println(element)
    }
}

fun main() {
    /*
    val tensor = DoublesTensor(
        Shape(3, 3),
        doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    )

    val slicedTensor = slice<Double>(tensor) {
        // from second to the last
        segment {
            all()
        }
        // all elements, equals to ":"
        // from 0 to the second last element
        segment {
            from(1)
        }
    }

    println(tensor)
    println(slicedTensor)

     */
}
