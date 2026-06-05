#pragma once

#include "phase1_types.h"

namespace smao::phase1 {

/**
 * Algorithm 4: AnisotropicDistancePrimitive
 *
 * Computes anisotropic squared distance:
 *   d_M^2(q, k) = (q - k)^T * M * (q - k)
 *
 * This is the ONLY distance function exposed to Phase 2.
 * All spatial indexing and field evaluation depend on this primitive.
 *
 * @param q Query point (d-dimensional)
 * @param k Key point (d-dimensional)
 * @param M Metric tensor (d x d)
 * @return Anisotropic squared distance
 */
f32 anisotropic_squared_distance(
    const VectorXf& q,
    const VectorXf& k,
    const MatrixXf& M
);

/**
 * Compute anisotropic squared distance (via whitened coordinates for numerical stability)
 *
 * Equivalent formulation using pre-computed whitening:
 *   d_M^2(q, k) = ||W*q - W*k||_2^2
 *
 * @param q_whitened Whitened query point W*q (d-dimensional)
 * @param k_whitened Whitened key point W*k (d-dimensional)
 * @return Anisotropic squared distance
 */
f32 anisotropic_squared_distance_whitened(
    const VectorXf& q_whitened,
    const VectorXf& k_whitened
);

}  // namespace smao::phase1
