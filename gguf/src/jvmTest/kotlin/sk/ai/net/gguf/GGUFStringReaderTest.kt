package sk.ai.net.gguf

import junit.framework.Assert.assertEquals
import kotlinx.io.asSource
import kotlinx.io.buffered
import org.junit.Test


class GGUFStringReaderTest {

    @Test
    fun testReadMetadataFields() {
        javaClass.getResourceAsStream("/test_experiment.gguf").use { inputStream ->

            val reader = GGUFReader(inputStream.asSource().buffered())

            // Verify the 'general.name' metadata is correct
            val modelName = reader.getString("general.name")
            assertEquals("general.name should match", "addition_op", modelName)

            // Verify the 'general.architecture' metadata is correct
            val architecture = reader.getString("general.architecture")
            assertEquals("general.architecture should match", "llama", architecture)
        }
    }
}
