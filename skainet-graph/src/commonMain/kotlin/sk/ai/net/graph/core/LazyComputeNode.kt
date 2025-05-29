package sk.ai.net.graph.core

/**
 * A compute node that lazily evaluates its result.
 *
 * This node caches the result of its evaluation, so that subsequent calls to
 * [evaluate] return the cached result instead of recomputing it. This can
 * significantly improve performance for nodes that are evaluated multiple times.
 *
 * @param T The type of data processed by this computation node.
 * @property node The underlying computation node.
 */
class LazyComputeNode<T>(private val node: ComputeNode<T>) : ComputeNode<T>() {
    /**
     * The cached result of the evaluation.
     */
    private var cachedResult: T? = null

    /**
     * Whether the result has been cached.
     */
    private var hasResult = false

    /**
     * Evaluates this node and returns the result.
     *
     * If the result has already been computed, returns the cached result.
     * Otherwise, computes the result, caches it, and returns it.
     *
     * @return The result of the computation.
     */
    override fun evaluate(): T {
        if (!hasResult) {
            cachedResult = node.evaluate()
            hasResult = true
        }
        return cachedResult as T
    }

    /**
     * Clears the cached result, forcing the node to recompute its result
     * the next time [evaluate] is called.
     */
    fun clearCache() {
        cachedResult = null
        hasResult = false
    }
}

/**
 * Creates a lazy version of a computation node.
 *
 * @param T The type of data processed by the computation node.
 * @return A lazy version of the computation node.
 */
fun <T> ComputeNode<T>.lazy(): LazyComputeNode<T> = LazyComputeNode(this)