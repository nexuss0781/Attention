/**
 * @file exact_decomposition.h
 * @brief Exact softmax-to-Gaussian decomposition
 *
 * Implements Theorem 1.1: Gaussian decomposition with floating-point error bounds.
 * Computes a_i = exp(||q_i||^2 / 2*sigma^2) and w_j = exp(||k_j||^2 / 2*sigma^2).
 */

#ifndef SMAO_EXACT_DECOMPOSITION_H
#define SMAO_EXACT_DECOMPOSITION_H

#include "types.h"

namespace smao {

/**
 * @brief Compute Kahan-compensated dot product
 *
 * Implements compensated summation for improved numerical accuracy.
 * Reduces floating-point error from O(n*eps) to O(eps).
 *
 * @param x First vector
 * @param y Second vector
 * @param n Vector dimension
 * @return Dot product with Kahan compensation
 */
f32 kahan_dot_product(const f32* x, const f32* y, size_t n);

/**
 * @brief Compute squared norm with Kahan compensation
 *
 * @param x Input vector
 * @param n Vector dimension
 * @return Squared L2 norm
 */
f32 kahan_squared_norm(const f32* x, size_t n);

/**
 * @brief Exact softmax-to-Gaussian decomposition
 *
 * Implements Algorithm 1: ExactDecomposition
 * Computes a_i, w_j, and sigma^2 according to Theorem 1.1
 *
 * @param q Query matrix (n x d, row-major)
 * @param k Key matrix (n x d, row-major)
 * @param n Number of tokens
 * @param d Head dimension
 * @param d_k Head dimension (for sigma^2 = sqrt(d_k))
 * @param epsilon Regularization parameter
 * @param log_clip_min Minimum log-space value
 * @param log_clip_max Maximum log-space value
 * @param query_scales Output: a_i scaling factors (size n)
 * @param key_weights Output: w_j scaling weights (size n)
 * @param sigma_squared Output: bandwidth parameter
 * @return Status code
 */
Status exact_decomposition(
    const f32* q,
    const f32* k,
    size_t n,
    size_t d,
    size_t d_k,
    f32 epsilon,
    f32 log_clip_min,
    f32 log_clip_max,
    f32* query_scales,
    f32* key_weights,
    f32* sigma_squared
);

/**
 * @brief Vectorized decomposition for aligned memory
 *
 * Optimized version that assumes 32-byte aligned memory
 * and uses SIMD when available.
 *
 * @param q Query matrix (n x d, 32-byte aligned)
 * @param k Key matrix (n x d, 32-byte aligned)
 * @param n Number of tokens
 * @param d Head dimension
 * @param d_k Head dimension
 * @param query_scales Output: a_i (32-byte aligned)
 * @param key_weights Output: w_j (32-byte aligned)
 * @param sigma_squared Output: bandwidth
 * @return Status code
 */
Status exact_decomposition_aligned(
    const f32* q,
    const f32* k,
    size_t n,
    size_t d,
    size_t d_k,
    f32* query_scales,
    f32* key_weights,
    f32* sigma_squared
);

} // namespace smao

#endif // SMAO_EXACT_DECOMPOSITION_H
