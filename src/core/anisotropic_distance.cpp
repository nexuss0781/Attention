/**
 * @file anisotropic_distance.cpp
 * @brief Implementation of anisotropic distance computation
 *
 * Implements Algorithm 4: AnisotropicDistancePrimitive
 * Computes d^2_M(q,k) = (q-k)^T M (q-k)
 */

#include "smao_phase1/core/anisotropic_distance.h"
#include <cmath>

namespace smao {

f32 anisotropic_distance_squared(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
) {
    if (q == nullptr || k == nullptr || metric == nullptr || d == 0) {
        return 0.0f;
    }
    
    // Compute delta = q - k
    std::vector<f32> delta(d);
    for (size_t i = 0; i < d; ++i) {
        delta[i] = q[i] - k[i];
    }
    
    // Compute d^2_M = delta^T * M * delta
    // First compute M * delta
    std::vector<f32> m_delta(d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            m_delta[i] += metric[i * d + j] * delta[j];
        }
    }
    
    // Then compute delta^T * (M * delta)
    f32 dist_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dist_sq += delta[i] * m_delta[i];
    }
    
    // Ensure non-negative (numerical errors might make it slightly negative)
    if (dist_sq < 0.0f && dist_sq > -1e-6f) {
        dist_sq = 0.0f;
    }
    
    return dist_sq;
}

f32 anisotropic_distance(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
) {
    f32 dist_sq = anisotropic_distance_squared(q, k, metric, d);
    return std::sqrt(dist_sq);
}

Status anisotropic_distance_batch(
    const f32* q,
    const f32* keys,
    size_t m,
    size_t d,
    const f32* metric,
    f32* distances
) {
    if (q == nullptr || keys == nullptr || metric == nullptr || distances == nullptr) {
        return Status::InvalidInput;
    }
    if (m == 0 || d == 0) {
        return Status::InvalidInput;
    }
    
    for (size_t j = 0; j < m; ++j) {
        const f32* k = keys + j * d;
        distances[j] = anisotropic_distance_squared(q, k, metric, d);
    }
    
    return Status::OK;
}

f32 anisotropic_distance_eigen(
    const VectorXf& q,
    const VectorXf& k,
    const MatrixXf& metric
) {
    VectorXf delta = q - k;
    f32 dist_sq = delta.transpose() * metric * delta;
    return dist_sq;
}

bool verify_anisotropic_consistency(
    const f32* q,
    const f32* k,
    const f32* metric,
    const f32* whitening_w,
    f32 sigma_squared,
    size_t d,
    f32* relative_error
) {
    if (q == nullptr || k == nullptr || metric == nullptr || 
        whitening_w == nullptr || relative_error == nullptr) {
        return false;
    }
    if (d == 0 || sigma_squared <= 0.0f) {
        return false;
    }
    
    // Compute anisotropic kernel: K_M(q,k) = exp(-(q-k)^T M (q-k) / 2*sigma^2)
    f32 dist_sq_m = anisotropic_distance_squared(q, k, metric, d);
    f32 kernel_anisotropic = std::exp(-dist_sq_m / (2.0f * sigma_squared));
    
    // Compute whitened coordinates: q_tilde = W*q, k_tilde = W*k
    std::vector<f32> q_tilde(d, 0.0f);
    std::vector<f32> k_tilde(d, 0.0f);
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            q_tilde[i] += whitening_w[i * d + j] * q[j];
            k_tilde[i] += whitening_w[i * d + j] * k[j];
        }
    }
    
    // Compute isotropic kernel: K_iso(q_tilde, k_tilde) = exp(-||q_tilde - k_tilde||^2 / 2*sigma^2)
    f32 dist_sq_iso = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        f32 diff = q_tilde[i] - k_tilde[i];
        dist_sq_iso += diff * diff;
    }
    f32 kernel_isotropic = std::exp(-dist_sq_iso / (2.0f * sigma_squared));
    
    // Compute relative error
    f32 abs_diff = std::abs(kernel_anisotropic - kernel_isotropic);
    f32 max_val = std::max(std::abs(kernel_anisotropic), std::abs(kernel_isotropic));
    
    if (max_val < 1e-10f) {
        *relative_error = abs_diff;
    } else {
        *relative_error = abs_diff / max_val;
    }
    
    // Return true if relative error is within tolerance
    constexpr f32 TOLERANCE = 1e-6f;
    return *relative_error < TOLERANCE;
}

} // namespace smao
