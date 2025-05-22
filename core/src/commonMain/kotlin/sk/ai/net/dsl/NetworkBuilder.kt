package sk.ai.net.dsl

import sk.ai.net.nn.activations.ActivationsWrapperModule
import sk.ai.net.Shape
import sk.ai.net.Tensor
import sk.ai.net.nn.Flatten
import sk.ai.net.nn.Input
import sk.ai.net.nn.Linear
import sk.ai.net.nn.Conv2d
import sk.ai.net.nn.Module
import sk.ai.net.nn.topology.MLP


// DSL Marker to restrict the DSL to its intended scope
@DslMarker
annotation class NetworkDsl

@NetworkDsl
fun network(content: NeuralNetworkDsl.() -> Unit) = NeuralNetworkDslImpl()
    .apply(content)
    .create()

@NetworkDsl
interface NetworkDslItem

@NetworkDsl
interface NeuralNetworkDsl : NetworkDslItem {
    fun input(inputSize: Int, id: String = "")

    fun flatten(id: String = "", content: FLATTEN.() -> Unit = {})

    fun conv2d(id: String = "", content: CONV2D.() -> Unit = {})

    fun dense(outputDimension: Int, id: String = "", content: DENSE.() -> Unit = {})
}

@NetworkDsl
interface DENSE : NetworkDslItem {
    var activation: (Tensor) -> Tensor
    fun weights(initBlock: (Shape) -> Tensor)
    fun bias(initBlock: (Shape) -> Tensor)
}

@NetworkDsl
interface FLATTEN : NetworkDslItem {
    var startDim: Int
    var endDim: Int
}

@NetworkDsl
interface CONV2D : NetworkDslItem {
    var outChannels: Int
    var kernelSize: Int
    var stride: Int
    var padding: Int
}


private fun getDefaultName(id: String, s: String, size: Int): String {
    if (id.isNotEmpty()) return id
    return "$s-$size"
}


fun createLinear(
    inFeatures: Int,
    outFeatures: Int,
    id: String,
    myInitWeights: Tensor? = null,
    myInitBias: Tensor? = null
): Linear {
    return when {
        myInitWeights != null && myInitBias != null ->
            Linear(
                inFeatures = inFeatures,
                outFeatures = outFeatures,
                name = id,
                initWeights = myInitWeights,
                initBias = myInitBias
            )

        myInitWeights != null ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id, initWeights = myInitWeights)

        myInitBias != null ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id, initBias = myInitBias)

        else ->
            Linear(inFeatures = inFeatures, outFeatures = outFeatures, name = id)
    }
}

class FlattenImpl(
    override var startDim: Int = 1,
    override var endDim: Int = -1,
    private val id: String
) : FLATTEN {
    fun create(): Module {
        return Flatten(startDim, endDim, id)
    }
}

class DenseImpl(
    private val inputDimension: Int, private val outputDimension: Int, private val id: String
) : DENSE {

    private var weightsValue: Tensor? = null
    private var biasValue: Tensor? = null
    private var _activation: (Tensor) -> Tensor = { tensor -> tensor }

    fun create(): List<Module> {

        val linear = createLinear(
            inFeatures = inputDimension,
            outFeatures = outputDimension,
            id = id,
            myInitWeights = weightsValue,
            myInitBias = biasValue
        )

        return listOf(
            linear,
            ActivationsWrapperModule(activation, "activation")
        )
    }

    override var activation: (Tensor) -> Tensor
        get() = _activation
        set(value) {
            _activation = value
        }

    override fun weights(initBlock: (Shape) -> Tensor) {
        weightsValue = initBlock(Shape(outputDimension, inputDimension))
    }

    override fun bias(initBlock: (Shape) -> Tensor) {
        biasValue = initBlock(Shape(outputDimension))
    }
}

class Conv2dImpl(
    private val inChannels: Int,
    override var outChannels: Int = 1,
    override var kernelSize: Int = 3,
    override var stride: Int = 1,
    override var padding: Int = 0,
    private val id: String
) : CONV2D {
    fun create(): Module = Conv2d(
        inChannels = inChannels,
        outChannels = outChannels,
        kernelSize = kernelSize,
        stride = stride,
        padding = padding,
        name = id
    )
}

private class NeuralNetworkDslImpl : NeuralNetworkDsl {

    val modules = mutableListOf<Module>()
    var lastDimension = 0

    fun create() = NetworkBuilder().add(*modules.toTypedArray()).build()
    override fun input(inputSize: Int, id: String) {
        lastDimension = inputSize
        modules.add(Input(Shape(inputSize), name = getDefaultName(id, "Input", modules.size)))
    }

    override fun flatten(id: String, content: FLATTEN.() -> Unit) {
        val impl = FlattenImpl(
            id = getDefaultName(id, "flatten", modules.size)
        )
        impl.content()
        modules += impl.create()
    }

    override fun conv2d(id: String, content: CONV2D.() -> Unit) {
        val impl = Conv2dImpl(
            inChannels = lastDimension,
            id = getDefaultName(id, "conv2d", modules.size)
        )
        impl.content()
        lastDimension = impl.outChannels
        modules += impl.create()
    }

    override fun dense(outputDimension: Int, id: String, content: DENSE.() -> Unit) {
        val inputDimension = lastDimension
        lastDimension = outputDimension
        val impl = DenseImpl(
            inputDimension = inputDimension,
            outputDimension = outputDimension,
            id = getDefaultName(id, "linear", modules.size)
        )
        impl.content()
        // dense layer consinst from linear module and activation function module (2 modules)
        modules += impl.create()
    }
}


@NetworkDsl
class NetworkBuilder {
    private val modules = mutableListOf<Module>()

    fun add(vararg modules: Module): NetworkBuilder {
        this.modules += modules.toList()
        return this
    }

    fun build(): Module = MLP(*modules.toTypedArray())
}
