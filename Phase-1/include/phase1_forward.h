#pragma once

#include "phase1_types.h"

namespace smao::phase1 {

/**
 * Algorithm 5: Phase1Forward
 *
 * Complete Phase 1 forward operator. Orchestrates all sub-algorithms to produce
 * the frozen gate outputs: whitened coordinates, scaling factors, metric geometry,
 * and anisotropic distance primitive.
 *
 * Input Contract:
 *   - Q, K, V: n x d embeddings, no NaN/Inf
 *   - L: d x d lower-triangular, diag(L) > 0
 *   - d_k: head dimension (typically = d)
 *
 * Output Contract (frozen gate):
 *   - Gaussian decomposition matches exact softmax to <1e-5 relative error (f32)
 *   - M is SPD with λ_min > 1e-6, κ < 1e4
 *   - Whitening isometry holds to <1e-6 relative error
 *   - Zero O(n²) auxiliary memory
 *   - Throughput: n=1e6, d=64 in <200ms
 *
 * @param Q Query embeddings (n x d)
 * @param K Key embeddings (n x d)
 * @param V Value embeddings (n x d_v) [stored but not processed in Phase 1]
 * @param L Learned metric factor (d x d, lower-triangular)
 * @param d_k Head dimension
 * @return Phase1Output structure with all computed values and status code
 */
Phase1Output phase1_forward(
    const MatrixXf& Q,
    const MatrixXf& K,
    const MatrixXf& V,
    const MatrixXf& L,
    u32 d_k
);

/**
 * Validate Phase 1 outputs against gate exit criteria
 *
 * @param output Phase1Output from phase1_forward
 * @param Q_original Original queries for reference computation
 * @param K_original Original keys for reference computation
 * @return Status code; Status::OK if all criteria satisfied
 */
Status validate_phase1_output(
    const Phase1Output& output,
    const MatrixXf& Q_original,
    const MatrixXf& K_original
);

}  // namespace smao::phase1
