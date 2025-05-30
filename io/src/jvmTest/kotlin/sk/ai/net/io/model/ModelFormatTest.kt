package sk.ai.net.io.model

import kotlinx.io.asSource
import kotlinx.io.buffered
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

class ModelFormatTest {

    @Test
    fun testGGUFModelFormat() {
        // Get the test GGUF file
        val ggufFile = javaClass.getResourceAsStream("/test_experiment.gguf")

        if (ggufFile == null) {
            println("[DEBUG_LOG] GGUF test file not found")
            return
        }

        try {
            // Create a GGUFModelFormat instance
            val modelFormat = GGUFModelFormat(ggufFile.asSource().buffered())

            // Test getMetadata
            val metadata = modelFormat.getMetadata()
            println("[DEBUG_LOG] GGUF metadata: $metadata")
            assertNotNull(metadata, "Metadata should not be null")

            // Test getTensorNames
            val tensorNames = modelFormat.getTensorNames()
            println("[DEBUG_LOG] GGUF tensor names: $tensorNames")
            assertNotNull(tensorNames, "Tensor names should not be null")

            if (tensorNames.isNotEmpty()) {
                // Test getTensor
                val tensorName = tensorNames.first()
                val tensor = modelFormat.getTensor(tensorName)
                println("[DEBUG_LOG] GGUF tensor $tensorName: $tensor")
                assertNotNull(tensor, "Tensor should not be null")

                // Test getAllTensors
                val tensors = modelFormat.getAllTensors()
                System.out.println("[DEBUG_LOG] GGUF tensors: ${tensors.keys}")
                assertNotNull(tensors, "Tensors should not be null")
                assertEquals(tensorNames.size, tensors.size, "Number of tensors should match")
            }
        } catch (e: Exception) {
            System.out.println("[DEBUG_LOG] Error in testGGUFModelFormat: ${e.message}")
            e.printStackTrace()
            fail("Error in testGGUFModelFormat: ${e.message}")
        }
    }

    @Test
    fun testModelFormatLoader() {
        // Get the test GGUF file
        val ggufFile = javaClass.getResourceAsStream("/test_experiment.gguf")

        if (ggufFile == null) {
            println("[DEBUG_LOG] GGUF test file not found")
            return
        }

        try {
            // Create a ModelFormatLoader instance
            val loader = ModelFormatLoader { ggufFile.asSource().buffered() }

            // Test getMetadata
            val metadata = loader.getMetadata()
            println("[DEBUG_LOG] ModelFormatLoader metadata: $metadata")
            System.out.println("[DEBUG_LOG] ModelFormatLoader metadata (System.out): $metadata")
            assertNotNull(metadata, "Metadata should not be null")

            // Test getTensorNames
            val tensorNames = loader.getTensorNames()
            println("[DEBUG_LOG] ModelFormatLoader tensor names: $tensorNames")
            assertNotNull(tensorNames, "Tensor names should not be null")

            if (tensorNames.isNotEmpty()) {
                // Test getTensor
                val tensorName = tensorNames.first()
                val tensor = loader.getTensor(tensorName)
                println("[DEBUG_LOG] ModelFormatLoader tensor $tensorName: $tensor")
                assertNotNull(tensor, "Tensor should not be null")

                // Test load
                var loadedTensors = 0
                kotlinx.coroutines.runBlocking {
                    loader.load { name, tensor ->
                        println("[DEBUG_LOG] ModelFormatLoader loaded tensor $name: $tensor")
                        loadedTensors++
                    }
                }
                assertEquals(
                    message = "Number of loaded tensors should match",
                    expected = tensorNames.size,
                    actual = loadedTensors
                )
            }
        } catch (e: Exception) {
            System.out.println("[DEBUG_LOG] Error in testModelFormatLoader: ${e.message}")
            e.printStackTrace()
            fail("Error in testModelFormatLoader: ${e.message}")
        }
    }

    @Test
    fun testModelFormatCreate() {
        // Get the test GGUF file
        val ggufFile = javaClass.getResourceAsStream("/test_experiment.gguf")

        if (ggufFile == null) {
            System.out.println("[DEBUG_LOG] GGUF test file not found")
            return
        }

        try {
            // Create a ModelFormat instance using the create method
            val modelFormat = ModelFormat.create(ggufFile.asSource().buffered())

            // Verify that it's a GGUFModelFormat instance
            assertTrue(modelFormat is GGUFModelFormat, "ModelFormat should be a GGUFModelFormat")

            // Test basic functionality
            val metadata = modelFormat.getMetadata()
            System.out.println("[DEBUG_LOG] ModelFormat.create metadata: $metadata")
            assertNotNull(metadata, "Metadata should not be null")

            val tensorNames = modelFormat.getTensorNames()
            System.out.println("[DEBUG_LOG] ModelFormat.create tensor names: $tensorNames")
            assertNotNull(tensorNames, "Tensor names should not be null")
        } catch (e: Exception) {
            System.out.println("[DEBUG_LOG] Error in testModelFormatCreate: ${e.message}")
            e.printStackTrace()
            fail("Error in testModelFormatCreate: ${e.message}")
        }
    }
}
