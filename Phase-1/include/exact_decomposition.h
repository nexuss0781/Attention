#pragma once

#include "phase1_types.h"
#include <cmath>

namespace smao::phase1 {

/**
 * Algorithm 1: ExactDecomposition
 *
 * Implements Theorem 1.1: Gaussian Decomposition with Floating-Point Error Bound
 *
 * For whitened coordinates Q, K and bandwidth sigma^2 = sqrt(d_k):
 *   exp(||q||_2^2 / (2*sigma^2)) = a[i]
 *   exp(||k||_2^2 / (2*sigma^2)) = w[j]
 *
 * Then: exp(<q, k> / sigma^2) = a[i] * w[j] * exp(-||q-k||_2^2 / (2*sigma^2))
 *
 * @param Q Query embeddings (n x d)
 * @param K Key embeddings (n x d)
 * @param d_k Head dimension (typically = d)
 * @param out_a Output: query scaling factors (n)
 * @param out_w Output: key scaling factors (n)
 * @param out_sigma2 Output: bandwidth parameter (scalar)
 * @return Status code
 */
Status exact_decomposition(
    const MatrixXf& Q,
    const MatrixXf& K,
    u32 d_k,
    VectorXf& out_a,
    VectorXf& out_w,
    f32& out_sigma2
);

}  // namespace smao::phase1
