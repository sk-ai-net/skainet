package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.Q4_0BlockTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.Q5_0BlockTensorData
import sk.ainet.lang.tensor.data.Q5_1BlockTensorData
import sk.ainet.lang.tensor.data.Q5_KBlockTensorData
import sk.ainet.lang.tensor.data.Q6_KBlockTensorData
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.data.Ternary2BitTensorData
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.tensor.storage.TensorStorage
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * SKEEP-003 M0-A3: every tensor reports a coherent `Format(dtype, encoding)`; a packed weight is
 * logically FP32 with its block encoding — never a `Byte`-typed tensor.
 */
class FormatTest {

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType> tensor(data: TensorData<*, *>, dtype: kotlin.reflect.KClass<T>) =
        VoidOpsTensor(data as TensorData<T, Any?>, dtype)

    @Test
    fun denseArrayDataReportsNoEncodingAndDenseFormat() {
        val f = DenseFloatArrayTensorData<FP32>(Shape(2, 2), FloatArray(4))
        assertNull(f.encoding)
        assertEquals(Format(FP32, TensorEncoding.Dense(4)), tensor(f, FP32::class).format)
        assertTrue(tensor(f, FP32::class).format.isDense)

        val i = DenseIntArrayTensorData<Int32>(Shape(3), IntArray(3))
        assertEquals(Format.dense(Int32), tensor(i, Int32::class).format)
    }

    @Test
    fun narrowFloatDataIsDenseAtTwoBytes() {
        assertEquals(Format(BF16, TensorEncoding.Dense(2)), tensor(Bf16DenseTensorData(Shape(4), ByteArray(8)), BF16::class).format)
        assertEquals(Format(FP16, TensorEncoding.Dense(2)), tensor(Fp16DenseTensorData(Shape(4), ByteArray(8)), FP16::class).format)
        // the physical encoding is reported even if the tensor is declared at a wider dtype
        assertEquals(TensorEncoding.Dense(2), tensor(Bf16DenseTensorData(Shape(4), ByteArray(8)), FP32::class).format.encoding)
    }

    @Test
    fun packedWeightsAreLogicallyFp32WithTheirBlockEncoding() {
        val cases: List<Pair<TensorData<*, *>, TensorEncoding>> = listOf(
            Q4_0BlockTensorData(Shape(2, 32), ByteArray(2 * 18)) to TensorEncoding.Q4_0,
            Q5_0BlockTensorData(Shape(2, 32), ByteArray(2 * 22)) to TensorEncoding.Q5_0,
            Q5_1BlockTensorData(Shape(2, 32), ByteArray(2 * 24)) to TensorEncoding.Q5_1,
            Q8_0BlockTensorData(Shape(2, 32), ByteArray(2 * 34)) to TensorEncoding.Q8_0,
            Q4_KBlockTensorData(Shape(1, 256), ByteArray(144)) to TensorEncoding.Q4_K,
            Q5_KBlockTensorData(Shape(1, 256), ByteArray(176)) to TensorEncoding.Q5_K,
            Q6_KBlockTensorData(Shape(1, 256), ByteArray(210)) to TensorEncoding.Q6_K,
            Ternary2BitTensorData.zeros(Shape(2, 8)) to TensorEncoding.TernaryPacked,
        )
        for ((data, enc) in cases) {
            assertEquals(enc, data.encoding, "TensorData.encoding of ${data::class.simpleName}")
            val fmt = tensor(data, FP32::class).format
            assertEquals(Format(FP32, enc), fmt, "Format of ${data::class.simpleName}")
            assertSame(FP32, fmt.dtype)
            assertEquals("Float32/${enc.name}", fmt.toString())
        }
    }

    @Test
    fun storageFormatPairsDtypeAndEncoding() {
        val s = TensorStorage(Shape(1, 256), FP32, TensorEncoding.Q4_K, BufferHandle.Borrowed(ByteArray(144)))
        assertEquals(Format(FP32, TensorEncoding.Q4_K), s.format)
        assertEquals(144L, s.format.physicalBytes(256))
        assertEquals(Format.dense(FP32), TensorStorage(Shape(2), FP32, TensorEncoding.Dense(4), BufferHandle.Borrowed(ByteArray(8))).format)
    }

    @Test
    fun nonConcreteWitnessHasNoFormat() {
        val t = VoidOpsTensor<DType, Float>(DenseFloatArrayTensorData<DType>(Shape(1), FloatArray(1)), DType::class)
        assertNull(t.formatOrNull)
        assertFailsWith<IllegalStateException> { t.format }
    }
}
