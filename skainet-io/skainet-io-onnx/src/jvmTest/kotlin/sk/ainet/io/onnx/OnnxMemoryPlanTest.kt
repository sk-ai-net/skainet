package sk.ainet.io.onnx

import onnx.GraphProto
import onnx.ModelProto
import onnx.StringStringEntryProto
import onnx.TensorProto
import pbandk.ByteArr
import pbandk.encodeToByteArray
import sk.ainet.io.JvmRandomAccessSource
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * #1169: footprint planning from an ONNX initializer table — including `external_data`, whose
 * lengths previously reported ~0 bytes for exactly the >2 GB models that do not fit.
 */
class OnnxMemoryPlanTest {

    private fun model(vararg tensors: TensorProto): ByteArray =
        ModelProto(irVersion = 8, producerName = "OnnxMemoryPlanTest", graph = GraphProto(name = "g", initializer = tensors.toList()))
            .encodeToByteArray()

    private fun <R> withReader(bytes: ByteArray, block: (StreamingOnnxReader) -> R): R {
        val f = Files.createTempFile("plan_model", ".onnx").toFile()
        f.deleteOnExit()
        f.writeBytes(bytes)
        return StreamingOnnxReader.open(JvmRandomAccessSource.open(f)).use(block)
    }

    @Test
    fun planInputPricesRawDataInitializers() {
        val bytes = model(
            TensorProto(name = "w", dims = listOf(4L, 8L), dataType = TensorProto.DataType.FLOAT.value, rawData = ByteArr(ByteArray(128))),
            TensorProto(name = "b", dims = listOf(8L), dataType = TensorProto.DataType.FLOAT.value, rawData = ByteArr(ByteArray(32))),
        )
        withReader(bytes) { reader ->
            val input = reader.planInput(modelName = "test.onnx")
            assertEquals(2, input.weights.size)
            assertNull(input.geometry, "ONNX carries no architecture metadata")
            assertEquals("onnx", input.architecture)
            val w = input.weights.first { it.name == "w" }
            assertEquals(32L, w.elementCount)
            assertEquals(128L, w.bytes)
            assertEquals(160L, input.weights.sumOf { it.bytes })
        }
    }

    @Test
    fun externalDataLengthIsPricedNotZero() {
        // A weight stored in a sibling file: no raw_data, external_data carries location/offset/length.
        val threeGiB = 3L * 1024 * 1024 * 1024
        val bytes = model(
            TensorProto(
                name = "big",
                dims = listOf(threeGiB / 4),
                dataType = TensorProto.DataType.FLOAT.value,
                externalData = listOf(
                    StringStringEntryProto(key = "location", value = "model.onnx_data"),
                    StringStringEntryProto(key = "offset", value = "0"),
                    StringStringEntryProto(key = "length", value = threeGiB.toString()),
                ),
                dataLocation = TensorProto.DataLocation.EXTERNAL,
            ),
        )
        withReader(bytes) { reader ->
            val t = reader.tensors.single()
            assertEquals("model.onnx_data", t.externalLocation)
            assertEquals(threeGiB, t.externalLength)
            assertEquals(threeGiB, t.estimatedBytesLong, "external length must be priced, in Long")
            assertEquals(Int.MAX_VALUE, t.estimatedBytes, "Int view clamps instead of wrapping negative")
            val input = reader.planInput(modelName = "big.onnx")
            assertEquals(threeGiB, input.weights.single().bytes)
        }
    }

    @Test
    fun externalDataWithoutLengthFallsBackToElementsTimesTypeSize() {
        val bytes = model(
            TensorProto(
                name = "ext",
                dims = listOf(1024L),
                dataType = TensorProto.DataType.FLOAT.value,
                externalData = listOf(StringStringEntryProto(key = "location", value = "model.onnx_data")),
                dataLocation = TensorProto.DataLocation.EXTERNAL,
            ),
        )
        withReader(bytes) { reader ->
            val t = reader.tensors.single()
            assertEquals(4096L, t.estimatedBytesLong, "no length entry: elements × type size")
            assertTrue(t.externalLocation != null)
        }
    }
}
