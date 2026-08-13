/*
 * @file exact_decomposition.cpp
 * @brief Stable and vectorized softmax-to-Gaussian scalar decomposition.
 */

#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/numerical_guards.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace smao {

f32 kahan_dot_product(const f32* x, const f32* y, size_t n) {
    if (x == nullptr || y == nullptr || n == 0) return 0.0f;
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
    if (x == nullptr || n == 0) return 0.0f;
    f32 lane0 = 0.0f;
    f32 lane1 = 0.0f;
    f32 lane2 = 0.0f;
    f32 lane3 = 0.0f;
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        const f32 x0 = x[i];
        const f32 x1 = x[i + 1];
        const f32 x2 = x[i + 2];
        const f32 x3 = x[i + 3];
        lane0 += x0 * x0;
        lane1 += x1 * x1;
        lane2 += x2 * x2;
        lane3 += x3 * x3;
    }
    for (; i < n; ++i) {
        const f32 value = x[i];
        lane0 += value * value;
    }
    const f64 sum = static_cast<f64>(lane0) + static_cast<f64>(lane1) +
                    static_cast<f64>(lane2) + static_cast<f64>(lane3);
    if (!std::isfinite(sum) || sum > static_cast<f64>(std::numeric_limits<f32>::max())) {
        return std::numeric_limits<f32>::infinity();
    }
    return static_cast<f32>(sum);
}

namespace {

bool multiplication_fits(size_t a, size_t b) {
    return a == 0 || b <= std::numeric_limits<size_t>::max() / a;
}

Status validate_decomposition_shape(
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
        sigma_squared == nullptr || n == 0 || d == 0 || d_k == 0 ||
        !multiplication_fits(n, d) ||
        n > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        d > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) {
        return Status::InvalidInput;
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0f ||
        !std::isfinite(log_clip_min) || !std::isfinite(log_clip_max) ||
        log_clip_min >= log_clip_max) {
        return Status::InvalidInput;
    }
    return Status::OK;
}

} // namespace

Status exact_decomposition_pair_prevalidated(
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
    const Status shape_status = validate_decomposition_shape(
        q, k, n, d, d_k, epsilon, log_clip_min, log_clip_max,
        query_scales, key_weights, sigma_squared);
    if (shape_status != Status::OK) return shape_status;

    *sigma_squared = std::sqrt(static_cast<f32>(d_k));
    if (!std::isfinite(*sigma_squared) || !(*sigma_squared > 0.0f)) return Status::Overflow;
    const f32 denominator = 2.0f * (*sigma_squared);

    // Compute both norm arrays in one pass over Q and K. This removes two
    // independent matrix reductions from the end-to-end forward path.
#pragma omp parallel for schedule(static)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        const size_t offset = static_cast<size_t>(row) * d;
        f32 q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        f32 k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
        size_t i = 0;
        for (; i + 3 < d; i += 4) {
            q0 += q[offset + i] * q[offset + i];
            q1 += q[offset + i + 1] * q[offset + i + 1];
            q2 += q[offset + i + 2] * q[offset + i + 2];
            q3 += q[offset + i + 3] * q[offset + i + 3];
            k0 += k[offset + i] * k[offset + i];
            k1 += k[offset + i + 1] * k[offset + i + 1];
            k2 += k[offset + i + 2] * k[offset + i + 2];
            k3 += k[offset + i + 3] * k[offset + i + 3];
        }
        for (; i < d; ++i) {
            q0 += q[offset + i] * q[offset + i];
            k0 += k[offset + i] * k[offset + i];
        }
        const f32 q_exponent = std::max(log_clip_min, std::min(log_clip_max,
            (q0 + q1 + q2 + q3) / denominator));
        const f32 k_exponent = std::max(log_clip_min, std::min(log_clip_max,
            (k0 + k1 + k2 + k3) / denominator));
        query_scales[row] = std::exp(q_exponent);
        key_weights[row] = std::exp(k_exponent);
    }

    return (Eigen::Map<const VectorXf>(query_scales, static_cast<Eigen::Index>(n)).allFinite() &&
            Eigen::Map<const VectorXf>(key_weights, static_cast<Eigen::Index>(n)).allFinite())
        ? Status::OK : Status::Overflow;
}

Status exact_decomposition_prevalidated(
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
    return exact_decomposition_pair_prevalidated(
        q, k, n, d, d_k, epsilon, log_clip_min, log_clip_max,
        query_scales, key_weights, sigma_squared);
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
    const Status shape_status = validate_decomposition_shape(
        q, k, n, d, d_k, epsilon, log_clip_min, log_clip_max,
        query_scales, key_weights, sigma_squared);
    if (shape_status != Status::OK) return shape_status;
    Status status = validate_no_nan_inf(q, n * d);
    if (status != Status::OK) return status;
    status = validate_no_nan_inf(k, n * d);
    if (status != Status::OK) return status;
    return exact_decomposition_pair_prevalidated(
        q, k, n, d, d_k, epsilon, log_clip_min, log_clip_max,
        query_scales, key_weights, sigma_squared);
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
