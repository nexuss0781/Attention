/**
 * @file anisotropic_distance.h
 * @brief Anisotropic distance computation primitive
 *
 * Implements Algorithm 4: AnisotropicDistancePrimitive
 * Computes d^2_M(q,k) = (q-k)^T M (q-k) using the learned metric.
 */

#ifndef SMAO_ANISOTROPIC_DISTANCE_H
#define SMAO_ANISOTROPIC_DISTANCE_H

#include "types.h"

namespace smao {

/**
 * @brief Compute anisotropic squared distance
 *
 * Implements Algorithm 4: AnisotropicDistancePrimitive
 * Computes d^2_M(q,k) = (q-k)^T M (q-k)
 *
 * This is the ONLY distance primitive exposed to Phase 2.
 * All spatial indexing operates on d^2_M, not Euclidean distance.
 *
 * @param q Query vector (size d)
 * @param k Key vector (size d)
 * @param metric Metric matrix M (d x d, row-major)
 * @param d Dimension
 * @return Anisotropic squared distance d^2_M(q,k)
 */
f32 anisotropic_distance_squared(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
);

/**
 * @brief Compute anisotropic distance (square root of squared distance)
 *
 * @param q Query vector (size d)
 * @param k Key vector (size d)
 * @param metric Metric matrix M (d x d, row-major)
 * @param d Dimension
 * @return Anisotropic distance d_M(q,k)
 */
f32 anisotropic_distance(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
);

/**
 * @brief Vectorized batch computation of anisotropic distances
 *
 * Computes distances between a single query and multiple keys.
 *
 * @param q Query vector (size d)
 * @param keys Key matrix (m x d, row-major)
 * @param m Number of keys
 * @param d Dimension
 * @param metric Metric matrix M (d x d, row-major)
 * @param distances Output distances (size m, pre-allocated)
 * @return Status code
 */
Status anisotropic_distance_batch(
    const f32* q,
    const f32* keys,
    size_t m,
    size_t d,
    const f32* metric,
    f32* distances
);

/**
 * @brief Compute anisotropic distance using Eigen types
 *
 * Convenience function for internal use with Eigen matrices.
 *
 * @param q Query vector (size d)
 * @param k Key vector (size d)
 * @param metric Metric matrix M (d x d)
 * @return Anisotropic squared distance
 */
f32 anisotropic_distance_eigen(
    const VectorXf& q,
    const VectorXf& k,
    const MatrixXf& metric
);

/**
 * @brief Verify anisotropic kernel consistency
 *
 * Checks that K_M(q,k) = exp(-d^2_M(q,k) / 2*sigma^2) equals
 * the isotropic Gaussian in whitened space.
 *
 * @param q Query vector
 * @param k Key vector
 * @param metric Metric M
 * @param whitening_w Whitening operator W
 * @param sigma_squared Bandwidth
 * @param d Dimension
 * @param relative_error Output relative error
 * @return true if consistency holds
 */
bool verify_anisotropic_consistency(
    const f32* q,
    const f32* k,
    const f32* metric,
    const f32* whitening_w,
    f32 sigma_squared,
    size_t d,
    f32* relative_error
);

} // namespace smao

#endif // SMAO_ANISOTROPIC_DISTANCE_H
