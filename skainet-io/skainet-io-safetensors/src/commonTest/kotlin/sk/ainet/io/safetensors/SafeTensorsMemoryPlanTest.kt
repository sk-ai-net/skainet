package sk.ainet.io.safetensors

import sk.ainet.io.RandomAccessSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * #1169: footprint planning from a safetensors header — sizes come from `data_offsets`, so they
 * are authoritative even for dtypes with no fixed per-element width.
 */
class SafeTensorsMemoryPlanTest {

    private class ByteSource(private val data: ByteArray) : RandomAccessSource {
        override val size: Long = data.size.toLong()
        override fun readAt(position: Long, length: Int): ByteArray =
            data.copyOfRange(position.toInt(), (position + length).toInt())
        override fun readAt(position: Long, buffer: ByteArray, offset: Int, length: Int): Int {
            val n = minOf(length, (size - position).toInt())
            data.copyInto(buffer, offset, position.toInt(), position.toInt() + n)
            return n
        }
        override fun close() {}
    }

    /** 8-byte LE header length + JSON header + zero payload — the whole format. */
    private fun file(headerJson: String, payloadBytes: Int): ByteArray {
        val header = headerJson.encodeToByteArray()
        val out = ByteArray(8 + header.size + payloadBytes)
        var len = header.size.toLong()
        for (i in 0 until 8) { out[i] = (len and 0xFF).toByte(); len = len shr 8 }
        header.copyInto(out, 8)
        return out
    }

    @Test
    fun planInputPricesFromDataOffsets() {
        val json = """{"w":{"dtype":"F32","shape":[4,8],"data_offsets":[0,128]},""" +
            """"h":{"dtype":"BF16","shape":[8],"data_offsets":[128,144]}}"""
        val reader = StreamingSafeTensorsReader.open(ByteSource(file(json, 144)))
        val input = reader.planInput(modelName = "test.safetensors")
        assertEquals(2, input.weights.size)
        assertNull(input.geometry, "safetensors carries no architecture metadata")
        assertEquals("safetensors", input.architecture)
        val w = input.weights.first { it.name == "w" }
        assertEquals(32L, w.elementCount)
        assertEquals(128L, w.bytes)
        assertEquals(w.bytes, w.residentBytes, "no form resolved: resident = as stored")
        assertEquals(144L, input.weights.sumOf { it.bytes })
    }

    @Test
    fun unknownDtypeIsPricedByItsOffsets() {
        // A quantized/unknown dtype has no per-element width — data_offsets still price it.
        val json = """{"q":{"dtype":"Q4_K_M","shape":[256],"data_offsets":[0,144]}}"""
        val reader = StreamingSafeTensorsReader.open(ByteSource(file(json, 144)))
        val input = reader.planInput(modelName = "q.safetensors")
        val q = input.weights.single()
        assertEquals(144L, q.bytes)
        assertEquals("Q4_K_M", q.format.encoding.name)
    }
}
