/*
 * @file anisotropic_distance.cpp
 * @brief Allocation-free anisotropic metric distance kernels.
 */

#include "smao_phase1/core/anisotropic_distance.h"

#include <cmath>
#include <limits>

namespace smao {

namespace {

bool finite_buffer(const f32* data, size_t count) {
    if (data == nullptr && count != 0) return false;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

} // namespace

f32 anisotropic_distance_squared(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
) {
    if (q == nullptr || k == nullptr || metric == nullptr || d == 0 ||
        !finite_buffer(q, d) || !finite_buffer(k, d) || !finite_buffer(metric, d * d)) {
        return std::numeric_limits<f32>::quiet_NaN();
    }

    f64 distance = 0.0;
    for (size_t i = 0; i < d; ++i) {
        const f64 delta_i = static_cast<f64>(q[i]) - static_cast<f64>(k[i]);
        f64 metric_delta_i = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const f64 delta_j = static_cast<f64>(q[j]) - static_cast<f64>(k[j]);
            metric_delta_i += static_cast<f64>(metric[i * d + j]) * delta_j;
        }
        distance += delta_i * metric_delta_i;
    }

    if (!std::isfinite(distance)) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    if (distance < 0.0) {
        const f64 tolerance = 1e-6 * std::max<f64>(1.0, std::abs(distance));
        if (distance >= -tolerance) return 0.0f;
        return std::numeric_limits<f32>::quiet_NaN();
    }
    if (distance > static_cast<f64>(std::numeric_limits<f32>::max())) {
        return std::numeric_limits<f32>::infinity();
    }
    return static_cast<f32>(distance);
}

f32 anisotropic_distance(
    const f32* q,
    const f32* k,
    const f32* metric,
    size_t d
) {
    const f32 squared = anisotropic_distance_squared(q, k, metric, d);
    return std::isfinite(squared) && squared >= 0.0f
        ? std::sqrt(squared)
        : std::numeric_limits<f32>::quiet_NaN();
}

Status anisotropic_distance_batch(
    const f32* q,
    const f32* keys,
    size_t m,
    size_t d,
    const f32* metric,
    f32* distances
) {
    if (q == nullptr || keys == nullptr || metric == nullptr || distances == nullptr ||
        m == 0 || d == 0 || !finite_buffer(q, d) ||
        !finite_buffer(keys, m * d) || !finite_buffer(metric, d * d)) {
        return Status::InvalidInput;
    }

    for (size_t j = 0; j < m; ++j) {
        distances[j] = anisotropic_distance_squared(q, keys + j * d, metric, d);
        if (!std::isfinite(distances[j])) return Status::Overflow;
    }
    return Status::OK;
}

f32 anisotropic_distance_eigen(
    const VectorXf& q,
    const VectorXf& k,
    const MatrixXf& metric
) {
    if (q.size() == 0 || q.size() != k.size() || metric.rows() != q.size() ||
        metric.cols() != q.size() || !q.allFinite() || !k.allFinite() || !metric.allFinite()) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    const VectorXf delta = q - k;
    const f32 distance = (delta.transpose() * metric * delta)(0, 0);
    return std::isfinite(distance) && distance >= 0.0f
        ? distance
        : std::numeric_limits<f32>::quiet_NaN();
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
    if (q == nullptr || k == nullptr || metric == nullptr || whitening_w == nullptr ||
        relative_error == nullptr || d == 0 || !std::isfinite(sigma_squared) || sigma_squared <= 0.0f ||
        !finite_buffer(q, d) || !finite_buffer(k, d) || !finite_buffer(metric, d * d) ||
        !finite_buffer(whitening_w, d * d)) {
        return false;
    }

    const f32 dist_sq_m = anisotropic_distance_squared(q, k, metric, d);
    if (!std::isfinite(dist_sq_m)) return false;

    f64 dist_sq_iso = 0.0;
    for (size_t i = 0; i < d; ++i) {
        f64 q_tilde_i = 0.0;
        f64 k_tilde_i = 0.0;
        for (size_t j = 0; j < d; ++j) {
            q_tilde_i += static_cast<f64>(whitening_w[i * d + j]) * q[j];
            k_tilde_i += static_cast<f64>(whitening_w[i * d + j]) * k[j];
        }
        const f64 delta = q_tilde_i - k_tilde_i;
        dist_sq_iso += delta * delta;
    }
    if (!std::isfinite(dist_sq_iso)) return false;

    const f64 kernel_m = std::exp(-static_cast<f64>(dist_sq_m) / (2.0 * sigma_squared));
    const f64 kernel_iso = std::exp(-dist_sq_iso / (2.0 * sigma_squared));
    const f64 difference = std::abs(kernel_m - kernel_iso);
    const f64 scale = std::max(std::abs(kernel_m), std::abs(kernel_iso));
    *relative_error = static_cast<f32>(scale > 1e-20 ? difference / scale : difference);
    return std::isfinite(*relative_error) && *relative_error <= 1e-5f;
}

} // namespace smao
