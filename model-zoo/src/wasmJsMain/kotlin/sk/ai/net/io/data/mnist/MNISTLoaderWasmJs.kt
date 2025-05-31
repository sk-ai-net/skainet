package sk.ai.net.io.data.mnist

import io.ktor.client.HttpClient
import io.ktor.client.engine.js.Js
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import io.ktor.client.call.body
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * WASM JS implementation of the MNIST loader.
 *
 * @property config The configuration for the MNIST loader.
 */
class MNISTLoaderWasmJs(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {

    /**
     * Downloads and caches a file.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray = withContext(Dispatchers.Default) {
        // In WASM JS, we don't have file system access, so we can't cache files
        // We'll just download the file every time
        println("Downloading file: $url")
        val data = downloadFile(url)

        // Note: In a real implementation, we would use a JS gzip library to decompress the data
        // For this example, we're assuming the server provides uncompressed data for WASM clients
        println("WASM JS implementation does not support gzip decompression. Assuming data is already decompressed.")

        return@withContext data
    }

    /**
     * Downloads a file from a URL.
     *
     * @param url The URL to download from.
     * @return The bytes of the file.
     */
    private suspend fun downloadFile(url: String): ByteArray {
        val client = HttpClient(Js) {
            // No plugins needed for basic functionality
        }

        try {
            val httpResponse: HttpResponse = client.get(url)
            return httpResponse.body()
        } finally {
            client.close()
        }
    }

    companion object {
        /**
         * Creates a new instance of MNISTLoaderWasmJs with the default configuration.
         *
         * @return A new instance of MNISTLoaderWasmJs.
         */
        fun create(): MNISTLoaderWasmJs {
            return MNISTLoaderWasmJs(MNISTLoaderConfig())
        }

        /**
         * Creates a new instance of MNISTLoaderWasmJs with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of MNISTLoaderWasmJs.
         */
        fun create(cacheDir: String): MNISTLoaderWasmJs {
            return MNISTLoaderWasmJs(MNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of MNISTLoaderWasmJs with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of MNISTLoaderWasmJs.
         */
        fun create(config: MNISTLoaderConfig): MNISTLoaderWasmJs {
            return MNISTLoaderWasmJs(config)
        }
    }
}
