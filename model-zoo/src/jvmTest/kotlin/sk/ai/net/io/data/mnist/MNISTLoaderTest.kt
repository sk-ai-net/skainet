package sk.ai.net.io.data.mnist

import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import java.io.File

/**
 * Tests for the MNIST loader.
 */
class MNISTLoaderTest {

    /**
     * Tests loading the MNIST training dataset.
     */
    @Test
    fun testLoadTrainingData() = runBlocking {
        // Create a temporary directory for caching
        val tempDir = createTempDir("mnist-test")
        try {
            // Create a loader with the temporary directory
            val loader = MNISTLoaderFactory.create(tempDir.absolutePath)
            
            // Load the training data
            val dataset = loader.loadTrainingData()
            
            // Verify the dataset
            assertNotNull(dataset)
            assertTrue(dataset.size > 0)
            assertEquals(60000, dataset.size) // MNIST training set has 60,000 images
            
            // Verify the first image
            val firstImage = dataset.images[0]
            assertNotNull(firstImage)
            assertEquals(MNISTConstants.IMAGE_PIXELS, firstImage.image.size)
            assertTrue(firstImage.label >= 0 && firstImage.label <= 9)
            
            // Verify that the cache files were created
            val trainImagesFile = File(tempDir, MNISTConstants.TRAIN_IMAGES_FILENAME.removeSuffix(".gz"))
            val trainLabelsFile = File(tempDir, MNISTConstants.TRAIN_LABELS_FILENAME.removeSuffix(".gz"))
            assertTrue(trainImagesFile.exists())
            assertTrue(trainLabelsFile.exists())
            
            // Load the data again to test caching
            val cachedDataset = loader.loadTrainingData()
            assertEquals(dataset.size, cachedDataset.size)
        } finally {
            // Clean up
            tempDir.deleteRecursively()
        }
    }

    /**
     * Tests loading the MNIST test dataset.
     */
    @Test
    fun testLoadTestData() = runBlocking {
        // Create a temporary directory for caching
        val tempDir = createTempDir("mnist-test")
        try {
            // Create a loader with the temporary directory
            val loader = MNISTLoaderFactory.create(tempDir.absolutePath)
            
            // Load the test data
            val dataset = loader.loadTestData()
            
            // Verify the dataset
            assertNotNull(dataset)
            assertTrue(dataset.size > 0)
            assertEquals(10000, dataset.size) // MNIST test set has 10,000 images
            
            // Verify the first image
            val firstImage = dataset.images[0]
            assertNotNull(firstImage)
            assertEquals(MNISTConstants.IMAGE_PIXELS, firstImage.image.size)
            assertTrue(firstImage.label >= 0 && firstImage.label <= 9)
            
            // Verify that the cache files were created
            val testImagesFile = File(tempDir, MNISTConstants.TEST_IMAGES_FILENAME.removeSuffix(".gz"))
            val testLabelsFile = File(tempDir, MNISTConstants.TEST_LABELS_FILENAME.removeSuffix(".gz"))
            assertTrue(testImagesFile.exists())
            assertTrue(testLabelsFile.exists())
            
            // Load the data again to test caching
            val cachedDataset = loader.loadTestData()
            assertEquals(dataset.size, cachedDataset.size)
        } finally {
            // Clean up
            tempDir.deleteRecursively()
        }
    }

    /**
     * Tests the subset method of the MNIST dataset.
     */
    @Test
    fun testDatasetSubset() = runBlocking {
        // Create a loader with the default configuration
        val loader = MNISTLoaderFactory.create()
        
        // Load the training data
        val dataset = loader.loadTrainingData()
        
        // Create a subset
        val subset = dataset.subset(0, 100)
        
        // Verify the subset
        assertEquals(100, subset.size)
        assertEquals(dataset.images[0], subset.images[0])
        assertEquals(dataset.images[99], subset.images[99])
    }

    /**
     * Tests the configuration of the MNIST loader.
     */
    @Test
    fun testLoaderConfiguration() {
        // Create a loader with a custom configuration
        val config = MNISTLoaderConfig(
            cacheDir = "custom-cache-dir",
            useCache = false
        )
        val loader = MNISTLoaderFactory.create(config)
        
        // Verify that the loader was created successfully
        assertNotNull(loader)
    }
}