package sk.ai.net.impl

import sk.ai.net.DataDescriptor
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.core.Slice
import sk.ai.net.core.TypedTensor
import kotlin.collections.map
import kotlin.math.exp
import kotlin.math.pow

data class DoublesTensor(override val shape: Shape, val elements: DoubleArray) : TypedTensor<Double> {
    constructor(shape: Shape, element: Double = 0.0) : this(
        shape,
        doubleArrayOf(shape.volume.toDouble(), element)
    )

    // Companion object (similar to static context in Java)
    companion object {
        val doubleDataDescriptor = BuiltInDoubleDataDescriptor()

    }

    override val dataDescriptor: DataDescriptor
        get() = doubleDataDescriptor

    internal fun index(indices: IntArray): Int {
        assert(
            { indices.size == shape.dimensions.size },
            { "`indices.size` must be ${shape.dimensions.size}: ${indices.size}" })
        return shape.dimensions.zip(indices).fold(0) { a, x ->
            assert({ 0 <= x.second && x.second < x.first }, { "Illegal index: indices = ${indices}, shape = $shape" })
            a * x.first + x.second
        }
    }

    override operator fun get(vararg indices: Int): Double {
        return elements[index(indices)]
    }

    override operator fun get(vararg ranges: Slice): Tensor {
        val intRanges = ranges.toList().map { s ->
            IntRange(s.startIndex.toInt(), s.endIndex.toInt() - 1)
        }.toTypedArray()
        return this.get(*intRanges)
    }

    override val allElements: List<Double>
        get() = elements.toList()


    override operator fun get(vararg ranges: IntRange): TypedTensor<Double> {
        val size = ranges.size
        val shape = ranges.map { x -> x.last - x.first + 1 }
        val reversedShape = shape.reversed()
        val indices = IntArray(size)
        val elements = DoubleArray(shape.fold(1, Int::times)) {
            var i = it
            var dimensionIndex = size - 1
            for (dimension in reversedShape) {
                indices[dimensionIndex] = i % dimension + ranges[dimensionIndex].first
                i /= dimension
                dimensionIndex--
            }
            get(*indices)
        }
        return DoublesTensor(Shape(*shape.toIntArray()), elements)
    }


    /**
     * Checks if two shapes can be broadcast together according to standard broadcasting rules.
     * 
     * Broadcasting rules:
     * 1. If the two arrays have different numbers of dimensions, the shape of the array with fewer dimensions
     *    is padded with ones on the left.
     * 2. If the shape of the two arrays does not match in any dimension, the array with shape equal to 1
     *    in that dimension is stretched to match the other shape.
     * 3. If in any dimension the sizes disagree and neither is equal to 1, an error is raised.
     * 
     * @param shape1 First shape
     * @param shape2 Second shape
     * @return The resulting broadcast shape if compatible, null otherwise
     */
    private fun canBroadcast(shape1: IntArray, shape2: IntArray): IntArray? {
        val resultDims = IntArray(maxOf(shape1.size, shape2.size))

        // Pad the shorter shape with ones on the left
        val paddedShape1 = IntArray(resultDims.size) { 1 }
        val paddedShape2 = IntArray(resultDims.size) { 1 }

        for (i in shape1.indices) {
            paddedShape1[paddedShape1.size - shape1.size + i] = shape1[i]
        }

        for (i in shape2.indices) {
            paddedShape2[paddedShape2.size - shape2.size + i] = shape2[i]
        }

        // Check if the shapes are compatible and determine the result shape
        for (i in resultDims.indices) {
            if (paddedShape1[i] == paddedShape2[i]) {
                resultDims[i] = paddedShape1[i]
            } else if (paddedShape1[i] == 1) {
                resultDims[i] = paddedShape2[i]
            } else if (paddedShape2[i] == 1) {
                resultDims[i] = paddedShape1[i]
            } else {
                // Incompatible shapes
                return null
            }
        }

        return resultDims
    }

    private inline fun commutativeBinaryOperation(
        tensor: DoublesTensor,
        operation: (Double, Double) -> Double
    ): TypedTensor<Double> {
        // If shapes are identical, perform element-wise operation directly
        if (shape == tensor.shape) {
            return DoublesTensor(shape, zipMap(elements, tensor.elements, operation))
        }

        // Check if shapes can be broadcast
        val broadcastShape = canBroadcast(shape.dimensions, tensor.shape.dimensions)
        require(broadcastShape != null) {
            "Cannot broadcast shapes ${shape} and ${tensor.shape} together"
        }

        // Create result tensor with broadcast shape
        val resultShape = Shape(*broadcastShape)
        val resultElements = DoubleArray(resultShape.volume)

        // Compute strides for each tensor
        val thisStrides = computeBroadcastStrides(shape.dimensions, broadcastShape)
        val otherStrides = computeBroadcastStrides(tensor.shape.dimensions, broadcastShape)
        val resultStrides = computeStrides(broadcastShape)

        // Perform the operation with broadcasting
        for (i in 0 until resultShape.volume) {
            val resultIndices = unravelIndex(i, broadcastShape, resultStrides)

            val thisIndex = getIndexWithBroadcast(resultIndices, shape.dimensions, thisStrides)
            val otherIndex = getIndexWithBroadcast(resultIndices, tensor.shape.dimensions, otherStrides)

            resultElements[i] = operation(elements[thisIndex], tensor.elements[otherIndex])
        }

        return DoublesTensor(resultShape, resultElements)
    }

    /**
     * Computes strides for broadcasting a shape to a target shape.
     */
    private fun computeBroadcastStrides(originalShape: IntArray, targetShape: IntArray): IntArray {
        val originalStrides = computeStrides(originalShape)
        val result = IntArray(targetShape.size)

        // Pad with zeros for dimensions that will be broadcast
        val offset = targetShape.size - originalShape.size
        for (i in 0 until offset) {
            result[i] = 0
        }

        // Copy strides for existing dimensions, set to 0 for dimensions with size 1 (to be broadcast)
        for (i in 0 until originalShape.size) {
            val targetIndex = i + offset
            result[targetIndex] = if (originalShape[i] == 1) 0 else originalStrides[i]
        }

        return result
    }

    /**
     * Gets the index in the original array for a given set of indices in the broadcast array.
     */
    private fun getIndexWithBroadcast(indices: IntArray, originalShape: IntArray, strides: IntArray): Int {
        var index = 0
        val offset = indices.size - originalShape.size

        for (i in 0 until originalShape.size) {
            val targetIndex = i + offset
            // For dimensions with size 1, use index 0
            val dimIndex = if (originalShape[i] == 1) 0 else indices[targetIndex]
            index += dimIndex * strides[targetIndex]
        }

        return index
    }

    private inline fun nonCommutativeBinaryOperation(
        tensor: DoublesTensor,
        operation: (Double, Double) -> Double,
        reverseOperation: (Double, Double) -> Double
    ): DoublesTensor {
        // If shapes are identical, perform element-wise operation directly
        if (shape == tensor.shape) {
            return DoublesTensor(shape, zipMap(elements, tensor.elements, operation))
        }

        // Check if shapes can be broadcast
        val broadcastShape = canBroadcast(shape.dimensions, tensor.shape.dimensions)
        require(broadcastShape != null) {
            "Cannot broadcast shapes ${shape} and ${tensor.shape} together"
        }

        // Create result tensor with broadcast shape
        val resultShape = Shape(*broadcastShape)
        val resultElements = DoubleArray(resultShape.volume)

        // Compute strides for each tensor
        val thisStrides = computeBroadcastStrides(shape.dimensions, broadcastShape)
        val otherStrides = computeBroadcastStrides(tensor.shape.dimensions, broadcastShape)
        val resultStrides = computeStrides(broadcastShape)

        // For non-commutative operations, we need to be careful about which operation to use
        // If this tensor is broadcast to match tensor's shape, we need to use the reverse operation
        // in some cases
        val thisIsBroadcast = shape.dimensions.size < tensor.shape.dimensions.size ||
                shape.dimensions.any { it == 1 }
        val otherIsBroadcast = tensor.shape.dimensions.size < shape.dimensions.size ||
                tensor.shape.dimensions.any { it == 1 }

        // Perform the operation with broadcasting
        for (i in 0 until resultShape.volume) {
            val resultIndices = unravelIndex(i, broadcastShape, resultStrides)

            val thisIndex = getIndexWithBroadcast(resultIndices, shape.dimensions, thisStrides)
            val otherIndex = getIndexWithBroadcast(resultIndices, tensor.shape.dimensions, otherStrides)

            // Use the appropriate operation based on which tensor is being broadcast
            resultElements[i] = if (thisIsBroadcast && !otherIsBroadcast) {
                // If only this tensor is being broadcast, use the reverse operation
                reverseOperation(tensor.elements[otherIndex], elements[thisIndex])
            } else {
                // Otherwise, use the normal operation
                operation(elements[thisIndex], tensor.elements[otherIndex])
            }
        }

        return DoublesTensor(resultShape, resultElements)
    }

    override operator fun plus(other: Tensor): Tensor {
        return commutativeBinaryOperation(other as DoublesTensor) { lhs, rhs -> lhs + rhs }
    }

    override operator fun plus(other: Double): Tensor {
        return commutativeBinaryOperation(
            DoublesTensor(Shape(1), other)
        ) { lhs, rhs -> lhs + rhs }
    }

    override fun plus(other: Int): Tensor {
        return commutativeBinaryOperation(
            DoublesTensor(Shape(1), other.toDouble())
        ) { lhs, rhs -> lhs + rhs }
    }

    override operator fun minus(other: Tensor): Tensor {
        return nonCommutativeBinaryOperation(
            other as DoublesTensor,
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }

    override operator fun minus(other: Double): TypedTensor<Double> {
        return nonCommutativeBinaryOperation(
            DoublesTensor(Shape(1), other),
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }

    override fun minus(other: Int): Tensor {
        return nonCommutativeBinaryOperation(
            DoublesTensor(Shape(1), other.toDouble()),
            { lhs, rhs -> lhs - rhs },
            { lhs, rhs -> rhs - lhs })
    }


    operator fun times(tensor: TypedTensor<Double>): TypedTensor<Double> {
        return commutativeBinaryOperation(tensor as DoublesTensor) { lhs, rhs -> lhs * rhs }
    }

    operator fun div(tensor: TypedTensor<Double>): TypedTensor<Double> {
        return nonCommutativeBinaryOperation(
            tensor as DoublesTensor,
            { lhs, rhs -> lhs / rhs },
            { lhs, rhs -> rhs / lhs })
    }

    operator fun times(scalar: Double): TypedTensor<Double> {
        return DoublesTensor(shape, elements.map { it * scalar }.toDoubleArray())
    }

    operator fun div(scalar: Double): TypedTensor<Double> {
        return DoublesTensor(shape, elements.map { it / scalar }.toDoubleArray())
    }

    override fun toString(): String {
        return when (shape.dimensions.size) {
            1 -> { // 1D tensor
                //println(shape)
                vectorToString()
            }

            2 -> { // 2D tensor
                //println(shape)
                matrixToString()
            }

            else -> "Tensor(${shape}, ${elements.contentToString()})" // higher dimensions
        }
    }

    private fun vectorToString(): String {
        return elements.joinToString(prefix = "[", postfix = "]")
    }

    private fun matrixToString(): String {
        val (rows, cols) = shape.dimensions
        return (0 until rows).joinToString(separator = "\n", prefix = "[\n", postfix = "\n]") { r ->
            (0 until cols).joinToString(prefix = " [", postfix = "]") { c ->
                elements[r * cols + c].toString()
            }
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DoublesTensor

        if (shape != other.shape) return false
        if (!elements.contentEquals(other.elements)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = shape.hashCode()
        result = 31 * result + elements.contentHashCode()
        return result
    }


    override fun matmul(other: Tensor): Tensor {
        // Scalar multiplication
        if (shape.dimensions.isEmpty() && other.shape.dimensions.isEmpty()) {
            return DoublesTensor(Shape(), doubleArrayOf(elements[0] * (other as DoublesTensor).elements[0]))
        }

        // Scalar and Vector multiplication (scalar is `this`)
        if (shape.dimensions.isEmpty()) {
            return DoublesTensor(
                other.shape,
                (other as DoublesTensor).elements.map { (it * elements[0]) }.toList().toDoubleArray()
            )
        }

        // Scalar and Vector multiplication (scalar is `other`)
        if (other.shape.dimensions.isEmpty()) {
            return DoublesTensor(
                shape,
                elements.map { it * (other as DoublesTensor).elements[0] }.toList().toDoubleArray()
            )
        }

        // Vector and Matrix multiplication
        if (shape.dimensions.size == 1 && other.shape.dimensions.size == 2) {
            if (shape.dimensions[0] != other.shape.dimensions[0]) {
                throw IllegalArgumentException("Incompatible shapes for vector-matrix multiplication: vector shape ${shape} and matrix shape ${other.shape}")
            }
            val result = DoubleArray(other.shape.dimensions[1]) { 0.0 }
            for (i in elements.indices) {
                for (j in 0 until other.shape.dimensions[1]) {
                    result[j] += elements[i] * (other as DoublesTensor).elements[i * other.shape.dimensions[1] + j]
                }
            }
            return DoublesTensor(Shape(other.shape.dimensions[1]), result)
        }

        // Matrix and Matrix multiplication
        if (shape.dimensions.size == 2 && other.shape.dimensions.size == 2) {
            if (shape.dimensions[1] != other.shape.dimensions[0]) {
                throw IllegalArgumentException("Incompatible shapes for matrix-matrix multiplication: first matrix shape ${shape} and second matrix shape ${other.shape}. Inner dimensions must match: ${shape.dimensions[1]} != ${other.shape.dimensions[0]}")
            }
            val newShape = Shape(shape.dimensions[0], other.shape.dimensions[1])
            val result = DoubleArray(newShape.volume) { 0.0 }
            for (i in 0 until shape.dimensions[0]) {
                for (j in 0 until other.shape.dimensions[1]) {
                    for (k in 0 until shape.dimensions[1]) {
                        result[i * newShape.dimensions[1] + j] += elements[i * shape.dimensions[1] + k] * (other as DoublesTensor).elements[k * other.shape.dimensions[1] + j]
                    }
                }
            }
            return DoublesTensor(newShape, result)
        }

        throw IllegalArgumentException("Unsupported tensor shapes for matrix multiplication: this.shape = ${shape} and other.shape = ${other.shape}. Supported combinations are: scalar-scalar, scalar-vector, vector-scalar, vector-matrix, and matrix-matrix.")
    }

    override fun t(): Tensor {
        // Ensure the tensor is 2D
        if (this.shape.dimensions.size != 2) {
            throw IllegalArgumentException("Transpose is only implemented for 2D tensors.")
        }

        // New shape with dimensions swapped
        val newShape = Shape(this.shape.dimensions[1], this.shape.dimensions[0])

        // Create a new elements array to hold the transposed elements
        val newElements = DoubleArray(this.elements.size)

        // Populate the new elements array with the transposed elements
        for (i in 0 until shape.dimensions[0]) { // Original rows
            for (j in 0 until shape.dimensions[1]) { // Original columns
                // Calculate the index in the original flat array and the new index in the transposed array
                val originalIndex = i * shape.dimensions[1] + j
                val newIndex = j * shape.dimensions[0] + i
                // Assign the transposed value
                newElements[newIndex] = this.elements[originalIndex]
            }
        }

        // Return a new tensor with the transposed shape and elements
        return DoublesTensor(newShape, newElements)
    }

    override fun relu(): Tensor =
        DoublesTensor(shape, elements.map { elem -> if (elem > 0) elem else 0.0 }.toDoubleArray())

    override fun softmax(): Tensor {
        // Apply softmax to the last dimension by default
        val lastDim = shape.dimensions.size - 1
        return if (lastDim >= 0) {
            softmax(lastDim)
        } else {
            // For scalar tensors, just return exp(x) / exp(x) = 1
            this
        }
    }

    override fun pow(tensor: Tensor): Tensor {
        require(shape == tensor.shape) {
            "Incompatible shapes for pow operation: this.shape = ${shape}, tensor.shape = ${tensor.shape}"
        }
        return DoublesTensor(shape, zipMap(elements, (tensor as DoublesTensor).elements) { a, b -> a.pow(b) })
    }

    override fun pow(scalar: Double): Tensor {
        return DoublesTensor(shape, elements.map { it.pow(scalar) }.toDoubleArray())
    }


    override fun sin(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sin(it) }.toDoubleArray())

    override fun cos(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cos(it) }.toDoubleArray())

    override fun tan(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.tan(it) }.toDoubleArray())

    override fun asin(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.asin(it) }.toDoubleArray())

    override fun acos(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.acos(it) }.toDoubleArray())

    override fun atan(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.atan(it) }.toDoubleArray())

    override fun sinh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sinh(it) }.toDoubleArray())

    override fun cosh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cosh(it) }.toDoubleArray())

    override fun tanh(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.tanh(it) }.toDoubleArray())

    override fun exp(): Tensor =
        DoublesTensor(shape, elements.map { exp(it) }.toDoubleArray())

    override fun log(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.ln(it) }.toDoubleArray())

    override fun sqrt(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.sqrt(it) }.toDoubleArray())

    override fun cbrt(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.cbrt(it) }.toDoubleArray())

    override fun sigmoid(): Tensor =
        DoublesTensor(shape, elements.map { (1.0 / exp(-it)) }.toDoubleArray())


    override fun ln(): Tensor =
        DoublesTensor(shape, elements.map { kotlin.math.ln(it) }.toDoubleArray())

    override fun flatten(startDim: Int, endDim: Int): Tensor {
        val dims = shape.dimensions.toMutableList()
        var s = if (startDim < 0) dims.size + startDim else startDim
        var e = if (endDim < 0) dims.size + endDim else endDim

        // handle tensors without batch dimension by prepending 1
        while (dims.size <= e) {
            dims.add(0, 1)
            s += 1
            e += 1
        }

        val flatSize = dims.subList(s, e + 1).fold(1) { acc, v -> acc * v }
        val newDims = dims.take(s) + flatSize + dims.drop(e + 1)
        return DoublesTensor(Shape(*newDims.toIntArray()), elements.copyOf())
    }

    fun computeStrides(dimensions: IntArray): IntArray {
        val strides = IntArray(dimensions.size) { 1 }
        for (i in dimensions.lastIndex - 1 downTo 0) {
            strides[i] = strides[i + 1] * dimensions[i + 1]
        }
        return strides
    }

    fun unravelIndex(index: Int, dimensions: IntArray, strides: IntArray): IntArray {
        var idx = index
        val indices = IntArray(dimensions.size)
        for (i in strides.indices) {
            indices[i] = idx / strides[i]
            idx %= strides[i]
        }
        return indices
    }


    override fun softmax(dim: Int): Tensor {
        val actualDim = if (dim < 0) shape.dimensions.size + dim else dim
        if (actualDim < 0 || actualDim >= shape.dimensions.size) {
            throw IllegalArgumentException("Dimension out of range")
        }

        val strides = computeStrides(shape.dimensions)

        // Create a unique key for each group of elements that share the same indices except for the specified dimension
        fun getGroupKey(indices: IntArray): Int {
            var key = 0
            for (i in indices.indices) {
                if (i != actualDim) {
                    key = key * shape.dimensions[i] + indices[i]
                }
            }
            return key
        }

        // Calculate the number of groups
        val numGroups = shape.volume / shape.dimensions[actualDim]

        // Find the maximum value for each group for numerical stability
        val maxValues = DoubleArray(numGroups) { Double.NEGATIVE_INFINITY }
        for (index in elements.indices) {
            val indices = unravelIndex(index, shape.dimensions, strides)
            val groupKey = getGroupKey(indices)
            maxValues[groupKey] = maxOf(maxValues[groupKey], elements[index])
        }

        // Compute the exponential of (element - max) for each element and the sum for each group
        val exps = DoubleArray(elements.size)
        val sumExps = DoubleArray(numGroups) { 0.0 }

        for (index in elements.indices) {
            val indices = unravelIndex(index, shape.dimensions, strides)
            val groupKey = getGroupKey(indices)
            val shiftedValue = elements[index] - maxValues[groupKey]
            val expValue = exp(shiftedValue)
            exps[index] = expValue
            sumExps[groupKey] += expValue
        }

        // Normalize by the sum of exponential to get softmax probabilities
        val softmaxElements = DoubleArray(elements.size)
        for (index in elements.indices) {
            val indices = unravelIndex(index, shape.dimensions, strides)
            val groupKey = getGroupKey(indices)
            softmaxElements[index] = exps[index] / sumExps[groupKey]
        }

        return DoublesTensor(shape, softmaxElements)
    }
}

fun DoublesTensor.prod(): Double = this.elements.fold(1.0) { acc, element -> acc * element }
