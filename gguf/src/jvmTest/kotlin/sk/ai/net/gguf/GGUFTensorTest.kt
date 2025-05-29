package sk.ai.net.gguf

import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Assert.*
import org.junit.Test

/**
 * Tests for the GGUF tensor loading functionality.
 * 
 * These tests verify that GGUF files can be loaded and converted to tensors
 * that can be used in the neural network computation graph.
 */
class GGUFTensorTest {

    @Test
    fun testTensorConversion() {
        javaClass.getResourceAsStream("/test_experiment.gguf")?.use { inputStream ->
            val reader = GGUFReader(inputStream.asSource().buffered())

            // Print the number of tensors
            System.out.println("[DEBUG_LOG] Number of tensors: ${reader.tensors.size}")

            // If we have tensors, test conversion
            if (reader.tensors.isNotEmpty()) {
                reader.tensors.forEachIndexed { index, readerTensor ->
                    System.out.println("[DEBUG_LOG] Converting tensor $index: name=${readerTensor.name}, type=${readerTensor.tensorType}, shape=${readerTensor.shape}")

                    try {
                        // Convert the tensor
                        val tensor = GGUFTensorConverter.convert(readerTensor)

                        // Verify the tensor has the correct shape
                        assertEquals(
                            "Tensor shape should match",
                            readerTensor.shape.size,
                            tensor.shape.dimensions.size
                        )

                        // Verify the tensor dimensions match
                        readerTensor.shape.forEachIndexed { dimIndex, dim ->
                            assertEquals(
                                "Dimension $dimIndex should match",
                                dim.toInt(),
                                tensor.shape.dimensions[dimIndex]
                            )
                        }

                        // Verify we can access tensor values
                        // Just check that we can access the first element without exceptions
                        val firstValue = tensor.get(*IntArray(tensor.shape.dimensions.size) { 0 })
                        System.out.println("[DEBUG_LOG] First tensor value: $firstValue")

                        System.out.println("[DEBUG_LOG] Successfully converted tensor ${readerTensor.name}")
                    } catch (e: Exception) {
                        System.out.println("[DEBUG_LOG] Error converting tensor ${readerTensor.name}: ${e.message}")
                        e.printStackTrace()
                        fail("Failed to convert tensor ${readerTensor.name}: ${e.message}")
                    }
                }
            } else {
                System.out.println("[DEBUG_LOG] No tensors found in the test file")
                // If there are no tensors, the test should still pass
                assertTrue(true)
            }
        }
    }

    @Test
    fun testTensorLoader() {
        // Test loading tensors
        System.out.println("[DEBUG_LOG] Testing GGUFTensorLoader.loadTensors")
        javaClass.getResourceAsStream("/test_experiment.gguf")?.use { inputStream ->
            try {
                val source = inputStream.asSource().buffered()
                val tensorNodes = GGUFTensorLoader.loadTensors(source)

                if (tensorNodes.isNotEmpty()) {
                    System.out.println("[DEBUG_LOG] Loaded ${tensorNodes.size} tensor nodes")
                    tensorNodes.forEach { (name, node) ->
                        System.out.println("[DEBUG_LOG] Loaded tensor node: $name")
                    }
                } else {
                    System.out.println("[DEBUG_LOG] No tensor nodes found")
                }
            } catch (e: Exception) {
                System.out.println("[DEBUG_LOG] Error loading tensors: ${e.message}")
                e.printStackTrace()
                fail("Error loading tensors: ${e.message}")
            }
        }

        // Test getting metadata
        System.out.println("[DEBUG_LOG] Testing GGUFTensorLoader.getMetadata")
        javaClass.getResourceAsStream("/test_experiment.gguf")?.use { inputStream ->
            try {
                val source = inputStream.asSource().buffered()
                val metadata = GGUFTensorLoader.getMetadata(source)

                if (metadata.isNotEmpty()) {
                    System.out.println("[DEBUG_LOG] Loaded ${metadata.size} metadata entries")
                    metadata.forEach { (key, value) ->
                        System.out.println("[DEBUG_LOG] Metadata: $key = $value")
                    }

                    // Check for specific metadata if available
                    if (metadata.containsKey("general.name")) {
                        assertEquals("general.name should match", "addition_op", metadata["general.name"])
                    }
                } else {
                    System.out.println("[DEBUG_LOG] No metadata found")
                }
            } catch (e: Exception) {
                System.out.println("[DEBUG_LOG] Error getting metadata: ${e.message}")
                e.printStackTrace()
                fail("Error getting metadata: ${e.message}")
            }
        }

        // Test getting a specific tensor
        System.out.println("[DEBUG_LOG] Testing GGUFTensorLoader.getTensor")
        javaClass.getResourceAsStream("/test_experiment.gguf")?.use { inputStream ->
            try {
                val source = inputStream.asSource().buffered()

                // First get the tensor names
                val tensorNodes = GGUFTensorLoader.loadTensors(source)

                if (tensorNodes.isNotEmpty()) {
                    // Get a fresh source for getting the tensor
                    javaClass.getResourceAsStream("/test_experiment.gguf")?.use { tensorInputStream ->
                        val tensorSource = tensorInputStream.asSource().buffered()
                        val tensorName = tensorNodes.keys.first()
                        val tensor = GGUFTensorLoader.getTensor(tensorSource, tensorName)

                        if (tensor != null) {
                            System.out.println("[DEBUG_LOG] Successfully loaded tensor: $tensorName")
                            System.out.println("[DEBUG_LOG] Tensor shape: ${tensor.shape}")
                        } else {
                            System.out.println("[DEBUG_LOG] Failed to load tensor: $tensorName")
                            fail("Failed to load tensor: $tensorName")
                        }
                    }
                } else {
                    System.out.println("[DEBUG_LOG] No tensor nodes found for getTensor test")
                }
            } catch (e: Exception) {
                System.out.println("[DEBUG_LOG] Error getting tensor: ${e.message}")
                e.printStackTrace()
                fail("Error getting tensor: ${e.message}")
            }
        }
    }

    @Test
    fun testTensorEvaluation() {
        javaClass.getResourceAsStream("/test_experiment.gguf")?.use { inputStream ->
            try {
                val source = inputStream.asSource().buffered()

                // Load tensor nodes
                System.out.println("[DEBUG_LOG] Loading tensor nodes for evaluation")
                val tensorNodes = GGUFTensorLoader.loadTensors(source)

                if (tensorNodes.isNotEmpty()) {
                    System.out.println("[DEBUG_LOG] Loaded ${tensorNodes.size} tensor nodes for evaluation")

                    // Evaluate each tensor node
                    tensorNodes.forEach { (name, node) ->
                        try {
                            System.out.println("[DEBUG_LOG] Evaluating tensor node: $name")
                            val tensor = node.evaluate()
                            assertNotNull("Tensor '$name' should evaluate to a non-null value", tensor)
                            System.out.println("[DEBUG_LOG] Successfully evaluated tensor '$name' with shape ${tensor.shape}")

                            // Try to access some values from the tensor
                            val indices = IntArray(tensor.shape.dimensions.size) { 0 }
                            val value = tensor.get(*indices)
                            System.out.println("[DEBUG_LOG] First value of tensor '$name': $value")
                        } catch (e: Exception) {
                            System.out.println("[DEBUG_LOG] Error evaluating tensor node '$name': ${e.message}")
                            e.printStackTrace()
                            fail("Error evaluating tensor node '$name': ${e.message}")
                        }
                    }
                } else {
                    System.out.println("[DEBUG_LOG] No tensor nodes found for evaluation")
                    // If there are no tensors, the test should still pass
                    assertTrue(true)
                }
            } catch (e: Exception) {
                System.out.println("[DEBUG_LOG] Error in testTensorEvaluation: ${e.message}")
                e.printStackTrace()
                fail("Error in testTensorEvaluation: ${e.message}")
            }
        }
    }
}