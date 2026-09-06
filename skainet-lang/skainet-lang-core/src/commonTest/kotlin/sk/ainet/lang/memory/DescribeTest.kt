package sk.ainet.lang.memory

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.Q4_KBlockTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.BufferHandle
import sk.ainet.lang.tensor.storage.Placement
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.tensor.storage.TensorStorage
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DescribeTest {

    @Test
    fun tensorRendererPrintsIdFormatShapeKindAndPlaceholders() {
        val t = VoidOpsTensor(DenseFloatArrayTensorData<FP32>(Shape(2, 3), FloatArray(6)), FP32::class)
        val anon = t.describe().split(" · ")
        assertEquals(7, anon.size)
        assertEquals("—", anon[0]); assertEquals("Float32/Dense(4B)", anon[1]); assertEquals("[2, 3]", anon[2])
        assertEquals("—", anon[4]); assertEquals("scope —", anon[5]); assertEquals("storage —", anon[6])

        t.id = TensorId(listOf("model", "layers", "blk.3", "attn"), "q_proj.weight")
        assertTrue(t.describe().startsWith("model.layers.blk.3.attn.q_proj.weight · Float32/Dense(4B) · [2, 3] · "))
    }

    @Test
    fun packedWeightRendersLogicalDtypeAndEncoding() {
        @Suppress("UNCHECKED_CAST")
        val q = VoidOpsTensor(Q4_KBlockTensorData(Shape(1, 256), ByteArray(144)) as TensorData<FP32, Float>, FP32::class)
        assertTrue(q.describe().contains(" · Float32/Q4_K · [1, 256] · "), q.describe())
    }

    @Test
    fun storageRendererShowsKindAndOrigin() {
        val mapped = TensorStorage(
            Shape(2048, 2048), FP32, TensorEncoding.Q4_K,
            BufferHandle.FileBacked("/models/model.gguf", 0x1A3F000L, 144L * 16384), Placement.MMAP_WEIGHTS,
        )
        assertEquals(
            "model.layers.blk.3.attn.q_proj.weight · Float32/Q4_K · [2048, 2048] · Mapped · /models/model.gguf @0x1a3f000 · scope — · storage —",
            mapped.describe(TensorId.parse("model.layers.blk.3.attn.q_proj.weight")),
        )
        val owned = TensorStorage(Shape(4), FP32, TensorEncoding.Dense(4), BufferHandle.Owned(ByteArray(16)))
        assertEquals("— · Float32/Dense(4B) · [4] · Owned · — · scope — · storage —", owned.describe())
    }
}
