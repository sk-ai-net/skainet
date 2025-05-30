package sk.ai.net.core.tensor.dsl

import sk.ai.net.graph.core.ComputeNode
import sk.ai.net.core.nn.ActivationsWrapperModule
import sk.ai.net.core.nn.Conv2d
import sk.ai.net.core.nn.Dropout
import sk.ai.net.core.nn.Flatten
import sk.ai.net.core.nn.Input
import sk.ai.net.core.nn.Linear
import sk.ai.net.core.nn.MaxPool2d
import sk.ai.net.core.nn.Module
import sk.ai.net.core.nn.topology.MLP
import sk.ai.net.core.tensor.shape.Shape

// DSL Marker to restrict the DSL to its intended scope
@DslMarker
annotation class NetworkDsl

/**
 * Creates a neural network using the DSL.
 *
 * @param T The type of data processed by the modules.
 * @param context The context that holds the data type information and operations.
 * @param content The DSL content.
 * @return The created neural network.
 */
@NetworkDsl
fun <T> network(context: Context<T>, content: NeuralNetworkDsl<T>.() -> Unit): Module<T> = NeuralNetworkDslImpl(context)
    .apply(content)
    .create()

@NetworkDsl
interface NetworkDslItem

@NetworkDsl
interface NeuralNetworkDsl<T> : NetworkDslItem {
    fun input(inputSize: Int, id: String = "")

    fun flatten(id: String = "", content: FLATTEN<T>.() -> Unit = {})

    fun conv2d(id: String = "", content: CONV2D<T>.() -> Unit = {})

    fun maxPool2d(id: String = "", content: MAXPOOL2D<T>.() -> Unit = {})

    fun dropout(id: String = "", content: DROPOUT<T>.() -> Unit = {})

    fun dense(outputDimension: Int, id: String = "", content: DENSE<T>.() -> Unit = {})
}

@NetworkDsl
interface DENSE<T> : NetworkDslItem {
    var activation: (T) -> T
    fun weights(initBlock: (Shape) -> T)
    fun bias(initBlock: (Shape) -> T)
}

@NetworkDsl
interface FLATTEN<T> : NetworkDslItem {
    var startDim: Int
    var endDim: Int
}

@NetworkDsl
interface CONV2D<T> : NetworkDslItem {
    var outChannels: Int
    var kernelSize: Int
    var stride: Int
    var padding: Int
}

@NetworkDsl
interface MAXPOOL2D<T> : NetworkDslItem {
    var kernelSize: Int
    var stride: Int
}

@NetworkDsl
interface DROPOUT<T> : NetworkDslItem {
    var p: Double
    var inplace: Boolean
}

private fun getDefaultName(id: String, s: String, size: Int): String {
    if (id.isNotEmpty()) return id
    return "$s-$size"
}

class FlattenImpl<T>(
    override var startDim: Int = 1,
    override var endDim: Int = -1,
    private val id: String,
    private val context: Context<T>
) : FLATTEN<T> {
    fun create(): Module<T> {
        return Flatten(startDim, endDim, id, context.flatten)
    }
}

class DenseImpl<T>(
    private val inputDimension: Int,
    private val outputDimension: Int,
    private val id: String,
    private val context: Context<T>
) : DENSE<T> {

    private var weightsValue: T? = null
    private var biasValue: T? = null
    private var _activation: (T) -> T = { it }

    fun create(): List<Module<T>> {
        // Create weights and bias if they haven't been set
        val weights = weightsValue ?: context.createTensor(Shape(outputDimension, inputDimension))
        val bias = biasValue ?: context.createTensor(Shape(outputDimension))

        val linear = Linear(
            weights = weights,
            bias = bias,
            matmul = context.matmul,
            add = context.add
        )

        return listOf(
            linear,
            ActivationsWrapperModule(activation, "activation")
        )
    }

    override var activation: (T) -> T
        get() = _activation
        set(value) {
            _activation = value
        }

    override fun weights(initBlock: (Shape) -> T) {
        weightsValue = initBlock(Shape(outputDimension, inputDimension))
    }

    override fun bias(initBlock: (Shape) -> T) {
        biasValue = initBlock(Shape(outputDimension))
    }
}

class Conv2dImpl<T>(
    private val inChannels: Int,
    override var outChannels: Int = 1,
    override var kernelSize: Int = 3,
    override var stride: Int = 1,
    override var padding: Int = 0,
    private val id: String,
    private val context: Context<T>
) : CONV2D<T> {
    fun create(): Module<T> = Conv2d(
        inChannels = inChannels,
        outChannels = outChannels,
        kernelSize = kernelSize,
        stride = stride,
        padding = padding,
        name = id,
        convolution = context.convolution
    )
}

class MaxPool2dImpl<T>(
    override var kernelSize: Int = 2,
    override var stride: Int = 2,
    private val id: String,
    private val context: Context<T>
) : MAXPOOL2D<T> {
    fun create(): Module<T> = MaxPool2d(
        kernelSize = kernelSize,
        stride = stride,
        name = id,
        maxPool = context.maxPool
    )
}

class DropoutImpl<T>(
    override var p: Double = 0.5,
    override var inplace: Boolean = false,
    private val id: String,
    private val context: Context<T>
) : DROPOUT<T> {
    fun create(): Module<T> = Dropout(
        p = p,
        inplace = inplace,
        name = id,
        dropout = context.dropout
    )
}

private class NeuralNetworkDslImpl<T>(private val context: Context<T>) : NeuralNetworkDsl<T> {

    val modules = mutableListOf<Module<T>>()
    var lastDimension = 0
    var inputNode: ComputeNode<T>? = null

    fun create(): Module<T> = MLP(*modules.toTypedArray())

    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        val input = Input(
            shape = Shape(inputSize),
            name = getDefaultName(id, "Input", modules.size),
            createTensor = context.createTensor
        )
        inputNode = input.createNode()
        modules.add(input)
    }

    override fun flatten(id: String, content: FLATTEN<T>.() -> Unit) {
        val impl = FlattenImpl(
            id = getDefaultName(id, "flatten", modules.size),
            context = context
        )
        impl.content()
        modules += impl.create()
    }

    override fun conv2d(id: String, content: CONV2D<T>.() -> Unit) {
        val impl = Conv2dImpl(
            inChannels = lastDimension,
            id = getDefaultName(id, "conv2d", modules.size),
            context = context
        )
        impl.content()
        lastDimension = impl.outChannels
        modules += impl.create()
    }

    override fun maxPool2d(id: String, content: MAXPOOL2D<T>.() -> Unit) {
        val impl = MaxPool2dImpl(
            id = getDefaultName(id, "maxPool2d", modules.size),
            context = context
        )
        impl.content()
        modules += impl.create()
    }

    override fun dropout(id: String, content: DROPOUT<T>.() -> Unit) {
        val impl = DropoutImpl(
            id = getDefaultName(id, "dropout", modules.size),
            context = context
        )
        impl.content()
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE<T>.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl(
            inputDimension = inputDimension,
            outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size),
            context = context
        )
        impl.content()
        // dense layer consists of linear module and activation function module (2 modules)
        modules += impl.create()
    }
}

@NetworkDsl
class NetworkBuilder<T> {
    private val modules = mutableListOf<Module<T>>()

    fun add(vararg modules: Module<T>): NetworkBuilder<T> {
        this.modules += modules.toList()
        return this
    }

    fun build(): Module<T> = MLP(*modules.toTypedArray())
}