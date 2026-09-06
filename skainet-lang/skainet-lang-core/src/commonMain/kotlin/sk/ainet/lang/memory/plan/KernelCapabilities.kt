package sk.ainet.lang.memory.plan

import sk.ainet.lang.memory.blockSpec
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * What a backend's kernels can actually be fed (#1109).
 *
 * The question [WeightFormResolver] needs and nothing could answer before: *which encodings can
 * this target compute a matmul on without help?* A backend that has no kernel for the format on
 * disk will still produce correct output — by dequantizing, on every forward pass — so the absence
 * is invisible until someone profiles it. Asking first turns that into a decision made once.
 *
 * Declared here rather than in `skainet-backend-api` because the resolver lives beside
 * [PlannerProfile] and the two are used together, while the backend modules are downstream. The
 * registry-backed implementation is over there, where the providers are.
 */
public interface KernelCapabilities {

    /**
     * Can this target feed a matmul on a weight encoded as [encoding], with FP32 activations?
     *
     * `null` means dense — the question is then whether an FP32 matmul kernel exists at all, which
     * for any real backend it does.
     */
    public fun canFeedMatmul(encoding: TensorEncoding?): Boolean

    /**
     * Does the kernel for [encoding] read input-block-major bytes?
     *
     * Every packed matmul kernel in the tree does (#973), so the default is "yes, if it is blocked
     * and we can feed it". A backend whose packed kernels read canonical order overrides this and
     * gets `AS_STORED` bytes instead of a pointless permutation.
     */
    public fun wantsKernelFeedOrder(encoding: TensorEncoding): Boolean =
        encoding.blockSpec != null && canFeedMatmul(encoding)

    public companion object {

        /** A target with dense kernels and nothing else — the conservative assumption. */
        public val DENSE_ONLY: KernelCapabilities = object : KernelCapabilities {
            override fun canFeedMatmul(encoding: TensorEncoding?): Boolean = encoding == null
        }

        /** A target that can feed anything. Useful in tests; true of no real backend. */
        public val EVERYTHING: KernelCapabilities = object : KernelCapabilities {
            override fun canFeedMatmul(encoding: TensorEncoding?): Boolean = true
        }
    }
}
