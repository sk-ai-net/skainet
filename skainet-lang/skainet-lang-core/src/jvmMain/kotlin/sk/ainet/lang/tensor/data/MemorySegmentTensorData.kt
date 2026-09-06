package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Marker interface for tensor data backed by a [MemorySegment].
 * Backends can check for this interface to use MemorySegment-based SIMD
 * operations (e.g. `FloatVector.fromMemorySegment`) instead of array-based ops.
 */
public interface MemorySegmentBackedData {
    /** The underlying off-heap memory segment. */
    public val segment: MemorySegment

    /** Byte offset into [segment] where this tensor's data starts. */
    public val segmentByteOffset: Long
}

private val JAVA_FLOAT_LE: ValueLayout.OfFloat =
    ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN)

/**
 * Off-heap, 64-byte-aligned tensor data backed by a [MemorySegment].
 *
 * This avoids GC pressure and enables direct use with the Vector API's
 * `FloatVector.fromMemorySegment` for SIMD-friendly access patterns.
 *
 * Two constructors:
 * - Primary: allocates a fresh segment via [arena].
 * - Secondary (slice): wraps an existing segment region (zero-copy view).
 */
public class MemorySegmentTensorData<T : DType> private constructor(
    initialShape: Shape,
    override val segment: MemorySegment,
    override val segmentByteOffset: Long,
    private val ownsArena: Boolean,
) : TensorData<T, Float>, MemorySegmentBackedData {
    /**
     * A dense view over the *same* off-heap bytes (SKEEP-003 §4.1 façade): the storage borrows this
     * data's [segment] — nothing is copied and a migrated kernel unwraps it once with
     * `SegmentStorage.segment()`.
     */
    override val view: sk.ainet.lang.memory.TensorView
        get() = sk.ainet.lang.memory.TensorView.dense(
            sk.ainet.lang.memory.SegmentStorage.borrow(
                if (segmentByteOffset == 0L) segment else segment.asSlice(segmentByteOffset),
            ),
            shape,
            sk.ainet.lang.types.FP32,
        )


    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()

    /** Total number of floats in this tensor. */
    public val volume: Int get() = shape.volume

    /** Byte length of the data region. */
    public val byteSize: Long get() = volume.toLong() * Float.SIZE_BYTES

    // ---- constructors ----

    /**
     * Allocates a new off-heap segment from [arena] with the given [alignment].
     */
    public constructor(
        shape: Shape,
        arena: Arena,
        alignment: Long = 64L,
    ) : this(
        initialShape = shape,
        segment = arena.allocate(
            shape.volume.toLong() * Float.SIZE_BYTES,
            alignment,
        ),
        segmentByteOffset = 0L,
        ownsArena = false,
    )

    /**
     * Wraps an existing [segment] region starting at [byteOffset] (zero-copy slice).
     */
    public constructor(
        shape: Shape,
        segment: MemorySegment,
        byteOffset: Long = 0L,
    ) : this(
        initialShape = shape,
        segment = segment,
        segmentByteOffset = byteOffset,
        ownsArena = false,
    )

    // ---- element access ----

    override fun get(vararg indices: Int): Float {
        val flat = calcFlatIndex(indices)
        return segment.get(JAVA_FLOAT_LE, segmentByteOffset + flat.toLong() * Float.SIZE_BYTES)
    }

    override fun set(vararg indices: Int, value: Float) {
        val flat = calcFlatIndex(indices)
        segment.set(JAVA_FLOAT_LE, segmentByteOffset + flat.toLong() * Float.SIZE_BYTES, value)
    }

    // ---- bulk copy ----

    override fun copyToFloatArray(): FloatArray {
        val result = FloatArray(volume)
        MemorySegment.copy(
            segment, JAVA_FLOAT_LE, segmentByteOffset,
            result, 0, volume,
        )
        return result
    }

    /**
     * Bulk-copy a [FloatArray] into this tensor's segment.
     */
    public fun copyFromFloatArray(src: FloatArray, srcOffset: Int = 0, length: Int = volume) {
        require(length <= volume) { "length $length exceeds volume $volume" }
        MemorySegment.copy(
            src, srcOffset,
            segment, JAVA_FLOAT_LE, segmentByteOffset,
            length,
        )
    }

    // ---- slicing ----

    /**
     * Returns a zero-copy view into this tensor's segment.
     * The returned view shares the same underlying memory.
     */
    public fun slice(flatOffset: Int, size: Int): MemorySegmentTensorData<T> {
        require(flatOffset >= 0 && flatOffset + size <= volume) {
            "Slice [$flatOffset, ${flatOffset + size}) out of bounds for volume $volume"
        }
        return MemorySegmentTensorData(
            initialShape = Shape(size),
            segment = segment,
            segmentByteOffset = segmentByteOffset + flatOffset.toLong() * Float.SIZE_BYTES,
            ownsArena = false,
        )
    }

    // ---- internals ----

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * A [TensorDataFactory] that produces [MemorySegmentTensorData] tensors,
 * keeping all data off-heap for SIMD-friendly access.
 *
 * Per-tensor segments are allocated from `Arena.ofAuto()` so the underlying
 * direct memory is reclaimed by the GC Cleaner once the wrapping tensor is
 * unreachable. A long-lived shared arena would have pinned every op-output
 * tensor allocated by `DefaultCpuOpsBase` for the factory's lifetime — on a
 * 30-layer Gemma 4 forward pass that piled up tens of GB of direct memory
 * monotonically and exhausted `-XX:MaxDirectMemorySize` regardless of cap.
 * Loaders that need explicit lifetime control should allocate their own
 * `Arena.ofShared()` and use the slice constructor of [MemorySegmentTensorData].
 */
public class MemorySegmentTensorDataFactory(
    private val alignment: Long = 64L,
) : TensorDataFactory, AutoCloseable {

    private fun <T : DType> allocate(shape: Shape): MemorySegmentTensorData<T> =
        MemorySegmentTensorData(shape, Arena.ofAuto(), alignment)

    // ---- TensorDataFactory ----

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                td.segment.fill(0)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume)
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for zeros: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val ones = FloatArray(shape.volume) { 1.0f }
                td.copyFromFloatArray(ones)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { 1 }
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for ones: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val arr = FloatArray(shape.volume) { value.toFloat() }
                td.copyFromFloatArray(arr)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { value.toInt() }
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("Unsupported dtype for full: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> randn(
        shape: Shape,
        dtype: KClass<T>,
        mean: Float,
        std: Float,
        random: Random,
    ): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val data = FloatArray(shape.volume)
                var hasSpare = false
                var spare = 0.0f
                for (i in data.indices) {
                    if (hasSpare) {
                        data[i] = spare * std + mean
                        hasSpare = false
                    } else {
                        val u1 = random.nextFloat()
                        val u2 = random.nextFloat()
                        val z0 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                            cos(2.0 * PI * u2.toDouble()).toFloat()
                        val z1 = sqrt(-2.0 * ln(u1.toDouble())).toFloat() *
                            kotlin.math.sin(2.0 * PI * u2.toDouble()).toFloat()
                        data[i] = z0 * std + mean
                        spare = z1
                        hasSpare = true
                    }
                }
                td.copyFromFloatArray(data)
                td as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("randn only supports floating point types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> uniform(
        shape: Shape,
        dtype: KClass<T>,
        min: Float,
        max: Float,
        random: Random,
    ): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val range = max - min
                val data = FloatArray(shape.volume) { random.nextFloat() * range + min }
                td.copyFromFloatArray(data)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { random.nextInt(min.toInt(), max.toInt()) }
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("uniform supports floating point and Int32 types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> init(
        shape: Shape,
        dtype: KClass<T>,
        generator: (indices: IntArray) -> V,
    ): TensorData<T, V> {
        val strides = IntArray(shape.dimensions.size)
        var stride = 1
        for (i in shape.dimensions.indices.reversed()) {
            strides[i] = stride
            stride *= shape.dimensions[i]
        }
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val data = FloatArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    generator(indices) as Float
                }
                td.copyFromFloatArray(data)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { flatIndex ->
                    val indices = IntArray(shape.dimensions.size)
                    var remaining = flatIndex
                    for (i in indices.indices) {
                        indices[i] = remaining / strides[i]
                        remaining %= strides[i]
                    }
                    generator(indices) as Int
                }
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("init supports floating point and Int32 types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> randomInit(
        shape: Shape,
        dtype: KClass<T>,
        generator: (random: Random) -> V,
        random: Random,
    ): TensorData<T, V> {
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                val data = FloatArray(shape.volume) { (generator(random) as Float) }
                td.copyFromFloatArray(data)
                td as TensorData<T, V>
            }
            Int32::class -> {
                val data = IntArray(shape.volume) { (generator(random) as Int) }
                DenseIntArrayTensorData<T>(shape, data) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("randomInit supports floating point and Int32 types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromFloatArray(
        shape: Shape,
        dtype: KClass<T>,
        data: FloatArray,
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        return when (dtype) {
            FP32::class, FP16::class -> {
                val td = allocate<T>(shape)
                td.copyFromFloatArray(data)
                td as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromFloatArray only supports floating point types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromIntArray(
        shape: Shape,
        dtype: KClass<T>,
        data: IntArray,
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        return when (dtype) {
            Int32::class -> {
                DenseIntArrayTensorData<T>(shape, data.copyOf()) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromIntArray only supports Int32 types: $dtype")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : DType, V> fromByteArray(
        shape: Shape,
        dtype: KClass<T>,
        data: ByteArray,
    ): TensorData<T, V> {
        require(data.size == shape.volume) {
            "Data size ${data.size} doesn't match shape volume ${shape.volume}"
        }
        return when (dtype) {
            Int8::class -> {
                // Fall back to dense byte representation
                val intData = IntArray(data.size) { data[it].toInt() }
                DenseIntArrayTensorData<T>(shape, intData) as TensorData<T, V>
            }
            else -> throw IllegalArgumentException("fromByteArray only supports Int8 types: $dtype")
        }
    }

    override fun close() {
        // No shared arena to close; per-tensor `Arena.ofAuto()` segments
        // are reclaimed by the GC Cleaner.
    }
}
