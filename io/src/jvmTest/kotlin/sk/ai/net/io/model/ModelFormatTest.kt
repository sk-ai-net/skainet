package sk.ai.net.io.model

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test
import org.junit.Assert.*
import sk.ai.net.graph.tensor.Tensor
import java.io.File

class ModelFormatTest {

    @Test
    fun testGGUFModelFormat() {
        // Get the test GGUF file
        val ggufFile = javaClass.getResourceAsStream("/test_experiment.gguf")
        
        if (ggufFile == null) {
            System.out.println("[DEBUG_LOG] GGUF test file not found")
            return
        }
        
        try {
            // Create a GGUFModelFormat instance
            val modelFormat = GGUFModelFormat(ggufFile.asSource().buffered())
            
            // Test getMetadata
            val metadata = modelFormat.getMetadata()
            System.out.println("[DEBUG_LOG] GGUF metadata: $metadata")
            assertNotNull("Metadata should not be null", metadata)
            
            // Test getTensorNames
            val tensorNames = modelFormat.getTensorNames()
            System.out.println("[DEBUG_LOG] GGUF tensor names: $tensorNames")
            assertNotNull("Tensor names should not be null", tensorNames)
            
            if (tensorNames.isNotEmpty()) {
                // Test getTensor
                val tensorName = tensorNames.first()
                val tensor = modelFormat.getTensor(tensorName)
                System.out.println("[DEBUG_LOG] GGUF tensor $tensorName: $tensor")
                assertNotNull("Tensor should not be null", tensor)
                
                // Test getAllTensors
                val tensors = modelFormat.getAllTensors()
                System.out.println("[DEBUG_LOG] GGUF tensors: ${tensors.keys}")
                assertNotNull("Tensors should not be null", tensors)
                assertEquals("Number of tensors should match", tensorNames.size, tensors.size)
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
            System.out.println("[DEBUG_LOG] GGUF test file not found")
            return
        }
        
        try {
            // Create a ModelFormatLoader instance
            val loader = ModelFormatLoader { ggufFile.asSource().buffered() }
            
            // Test getMetadata
            val metadata = loader.getMetadata()
            System.out.println("[DEBUG_LOG] ModelFormatLoader metadata: $metadata")
            assertNotNull("Metadata should not be null", metadata)
            
            // Test getTensorNames
            val tensorNames = loader.getTensorNames()
            System.out.println("[DEBUG_LOG] ModelFormatLoader tensor names: $tensorNames")
            assertNotNull("Tensor names should not be null", tensorNames)
            
            if (tensorNames.isNotEmpty()) {
                // Test getTensor
                val tensorName = tensorNames.first()
                val tensor = loader.getTensor(tensorName)
                System.out.println("[DEBUG_LOG] ModelFormatLoader tensor $tensorName: $tensor")
                assertNotNull("Tensor should not be null", tensor)
                
                // Test load
                var loadedTensors = 0
                loader.load { name, tensor ->
                    System.out.println("[DEBUG_LOG] ModelFormatLoader loaded tensor $name: $tensor")
                    loadedTensors++
                }
                assertEquals("Number of loaded tensors should match", tensorNames.size, loadedTensors)
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
            assertTrue("ModelFormat should be a GGUFModelFormat", modelFormat is GGUFModelFormat)
            
            // Test basic functionality
            val metadata = modelFormat.getMetadata()
            System.out.println("[DEBUG_LOG] ModelFormat.create metadata: $metadata")
            assertNotNull("Metadata should not be null", metadata)
            
            val tensorNames = modelFormat.getTensorNames()
            System.out.println("[DEBUG_LOG] ModelFormat.create tensor names: $tensorNames")
            assertNotNull("Tensor names should not be null", tensorNames)
        } catch (e: Exception) {
            System.out.println("[DEBUG_LOG] Error in testModelFormatCreate: ${e.message}")
            e.printStackTrace()
            fail("Error in testModelFormatCreate: ${e.message}")
        }
    }
}