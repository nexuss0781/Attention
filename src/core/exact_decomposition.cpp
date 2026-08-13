/*
 * @file exact_decomposition.cpp
 * @brief Stable softmax-to-Gaussian scalar decomposition.
 */

#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/numerical_guards.h"

#include <cmath>
#include <limits>

namespace smao {

f32 kahan_dot_product(const f32* x, const f32* y, size_t n) {
    if (x == nullptr || y == nullptr || n == 0) {
        return 0.0f;
    }
    f64 sum = 0.0;
    f64 compensation = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const f64 product = static_cast<f64>(x[i]) * static_cast<f64>(y[i]);
        const f64 corrected = product - compensation;
        const f64 updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
    }
    return static_cast<f32>(sum);
}

f32 kahan_squared_norm(const f32* x, size_t n) {
    if (x == nullptr || n == 0) {
        return 0.0f;
    }
    f64 sum = 0.0;
    f64 compensation = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const f64 square = static_cast<f64>(x[i]) * static_cast<f64>(x[i]);
        const f64 corrected = square - compensation;
        const f64 updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
    }
    if (!std::isfinite(sum) || sum > static_cast<f64>(std::numeric_limits<f32>::max())) {
        return std::numeric_limits<f32>::infinity();
    }
    return static_cast<f32>(sum);
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
    if (q == nullptr || k == nullptr || query_scales == nullptr || key_weights == nullptr ||
        sigma_squared == nullptr || n == 0 || d == 0 || d_k == 0) {
        return Status::InvalidInput;
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0f ||
        !std::isfinite(log_clip_min) || !std::isfinite(log_clip_max) ||
        log_clip_min >= log_clip_max) {
        return Status::InvalidInput;
    }

    Status status = validate_no_nan_inf(q, n * d);
    if (status != Status::OK) return status;
    status = validate_no_nan_inf(k, n * d);
    if (status != Status::OK) return status;

    *sigma_squared = std::sqrt(static_cast<f32>(d_k));
    if (!std::isfinite(*sigma_squared) || !(*sigma_squared > 0.0f)) {
        return Status::Overflow;
    }
    const f32 denominator = 2.0f * (*sigma_squared);

    auto compute_scale = [=](const f32* row, f32* result) -> Status {
        const f32 squared_norm = kahan_squared_norm(row, d);
        if (!std::isfinite(squared_norm)) {
            return Status::Overflow;
        }
        const f32 exponent = clip_exponent_arg(
            squared_norm / denominator, log_clip_min, log_clip_max);
        const f32 scale = safe_exp(exponent);
        if (!std::isfinite(scale) || !(scale > 0.0f)) {
            return Status::Overflow;
        }
        *result = scale;
        return Status::OK;
    };

    for (size_t i = 0; i < n; ++i) {
        status = compute_scale(q + i * d, query_scales + i);
        if (status != Status::OK) return status;
    }
    for (size_t j = 0; j < n; ++j) {
        status = compute_scale(k + j * d, key_weights + j);
        if (status != Status::OK) return status;
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
    return exact_decomposition(
        q, k, n, d, d_k, 1e-6f, -80.0f, 80.0f,
        query_scales, key_weights, sigma_squared);
}

} // namespace smao
