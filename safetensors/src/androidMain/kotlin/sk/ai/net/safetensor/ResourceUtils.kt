package sk.ai.net.safetensor

import android.content.Context
import android.content.res.AssetManager
import java.io.IOException

/**
 * Android implementation of ResourceUtils.
 */
actual object ResourceUtils {
    // Android context, must be set before using this class
    private var applicationContext: Context? = null

    /**
     * Sets the application context.
     * This must be called before using any other methods in this class.
     *
     * @param context The application context.
     */
    fun setContext(context: Context) {
        applicationContext = context.applicationContext
    }

    /**
     * Loads a resource as a byte array.
     *
     * @param path The path to the resource in the assets folder.
     * @return The resource as a byte array, or null if the resource was not found.
     */
    actual fun loadResourceAsBytes(path: String): ByteArray? {
        val context = applicationContext ?: return null
        
        return try {
            context.assets.open(path).use { it.readBytes() }
        } catch (e: IOException) {
            null
        }
    }
}