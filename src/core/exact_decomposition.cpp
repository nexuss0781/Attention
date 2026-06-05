/**
 * @file exact_decomposition.cpp
 * @brief Implementation of exact softmax-to-Gaussian decomposition
 *
 * Implements Algorithm 1: ExactDecomposition with Kahan-compensated
 * dot products for numerical accuracy.
 */

#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/numerical_guards.h"
#include <cmath>
#include <cstring>

namespace smao {

f32 kahan_dot_product(const f32* x, const f32* y, size_t n) {
    if (n == 0) return 0.0f;
    if (x == nullptr || y == nullptr) return 0.0f;

    // Kahan compensated summation
    f32 sum = 0.0f;
    f32 compensation = 0.0f;  // Running compensation for lost low-order bits

    for (size_t i = 0; i < n; ++i) {
        f32 product = x[i] * y[i];
        f32 compensated = product - compensation;
        f32 new_sum = sum + compensated;
        compensation = (new_sum - sum) - compensated;
        sum = new_sum;
    }

    return sum;
}

f32 kahan_squared_norm(const f32* x, size_t n) {
    if (n == 0) return 0.0f;
    if (x == nullptr) return 0.0f;

    // Kahan compensated summation of squares
    f32 sum = 0.0f;
    f32 compensation = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        f32 square = x[i] * x[i];
        f32 compensated = square - compensation;
        f32 new_sum = sum + compensated;
        compensation = (new_sum - sum) - compensated;
        sum = new_sum;
    }

    return sum;
}

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
) {
    // Validate inputs
    if (q == nullptr || k == nullptr) {
        return Status::InvalidInput;
    }
    if (query_scales == nullptr || key_weights == nullptr || sigma_squared == nullptr) {
        return Status::InvalidInput;
    }
    if (n == 0 || d == 0 || d_k == 0) {
        return Status::InvalidInput;
    }

    // Check for NaN/Inf in inputs
    Status nan_check = validate_no_nan_inf(q, n * d);
    if (nan_check != Status::OK) return nan_check;
    
    nan_check = validate_no_nan_inf(k, n * d);
    if (nan_check != Status::OK) return nan_check;

    // Compute sigma^2 = sqrt(d_k)
    *sigma_squared = std::sqrt(static_cast<f32>(d_k));
    
    // Compute denominator: 2 * sigma^2
    const f32 denom = 2.0f * (*sigma_squared);

    // Compute query scales a_i = exp(||q_i||^2 / 2*sigma^2)
    for (size_t i = 0; i < n; ++i) {
        const f32* q_row = q + i * d;
        
        // Compute squared norm with Kahan compensation
        f32 sq_norm = kahan_squared_norm(q_row, d);
        
        // Compute exponent argument
        f32 exp_arg = sq_norm / denom;
        
        // Clip to safe range and compute exp
        exp_arg = clip_exponent_arg(exp_arg, log_clip_min, log_clip_max);
        query_scales[i] = std::exp(exp_arg);
        
        // Check for overflow
        if (!std::isfinite(query_scales[i])) {
            return Status::Overflow;
        }
    }

    // Compute key weights w_j = exp(||k_j||^2 / 2*sigma^2)
    for (size_t j = 0; j < n; ++j) {
        const f32* k_row = k + j * d;
        
        // Compute squared norm with Kahan compensation
        f32 sq_norm = kahan_squared_norm(k_row, d);
        
        // Compute exponent argument
        f32 exp_arg = sq_norm / denom;
        
        // Clip to safe range and compute exp
        exp_arg = clip_exponent_arg(exp_arg, log_clip_min, log_clip_max);
        key_weights[j] = std::exp(exp_arg);
        
        // Check for overflow
        if (!std::isfinite(key_weights[j])) {
            return Status::Overflow;
        }
    }

    return Status::OK;
}

Status exact_decomposition_aligned(
    const f32* q,
    const f32* k,
    size_t n,
    size_t d,
    size_t d_k,
    f32* query_scales,
    f32* key_weights,
    f32* sigma_squared
) {
    // For aligned version, use default parameters
    constexpr f32 epsilon = 1e-6f;
    constexpr f32 log_clip_min = -80.0f;
    constexpr f32 log_clip_max = 80.0f;
    
    return exact_decomposition(
        q, k, n, d, d_k,
        epsilon, log_clip_min, log_clip_max,
        query_scales, key_weights, sigma_squared
    );
}

} // namespace smao
