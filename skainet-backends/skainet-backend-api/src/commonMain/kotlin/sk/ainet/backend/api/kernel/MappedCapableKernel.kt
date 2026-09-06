package sk.ainet.backend.api.kernel


/**
 * Marks a [ViewKernel] that serves its `BLOCKED_ROW_MAJOR` weight operand straight from off-heap
 * or mapped storage (a mmap'd GGUF page, a direct buffer) as well as from heap bytes — not just
 * the heap arm every packed kernel has. [FfmRowMajorMatmulKernel] and [JniRowMajorMatmulKernel]
 * are the two kernels that earn this today (#1189/#1192).
 *
 * [KernelDispatch.mappedServableEncodings] derives the encodings actually served from the
 * kernels registered with this marker (#1193) — the thing `StorageCapabilities.MAPPED_SERVABLE_DEFAULT`
 * (`skainet-lang-core`) and `KernelSupportMatrixTest.mappedTiers()`'s `native-jni-direct` row still
 * declare by hand, because neither can depend on this module (`skainet-backend-api`) without a
 * dependency cycle. `KernelSupportMatrixTest.generate_and_gate_support_matrix()` installs the
 * JVM-reachable mapped kernels and asserts the derived set against those hand-kept declarations,
 * so a kernel gaining or losing this marker without updating them fails CI instead of drifting
 * silently.
 */
public interface MappedCapableKernel
