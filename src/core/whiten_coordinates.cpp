/*
 * @file whiten_coordinates.cpp
 * @brief Implementation of coordinate whitening.
 */

#include "smao_phase1/core/whiten_coordinates.h"
#include "smao_phase1/core/numerical_guards.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>

namespace smao {

namespace {

bool finite_buffer(const f32* data, size_t count) {
    if (data == nullptr && count != 0) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

Status whiten_coordinates(
    const f32* x,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* x_whitened
) {
    if (x == nullptr || whitening_w == nullptr || x_whitened == nullptr || n == 0 || d == 0) {
        return Status::InvalidInput;
    }
    if (n > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        d > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) {
        return Status::InvalidInput;
    }
    if (!finite_buffer(x, n * d) || !finite_buffer(whitening_w, d * d)) {
        return Status::NaNInput;
    }

    using RowMajorMap = Eigen::Map<const MatrixXf>;
    using MutableRowMajorMap = Eigen::Map<MatrixXf>;
    const Eigen::Index rows = static_cast<Eigen::Index>(n);
    const Eigen::Index cols = static_cast<Eigen::Index>(d);
    RowMajorMap x_map(x, rows, cols);
    RowMajorMap w_map(whitening_w, cols, cols);
    MutableRowMajorMap output_map(x_whitened, rows, cols);
    output_map.noalias() = x_map * w_map;

    return finite_buffer(x_whitened, n * d) ? Status::OK : Status::Overflow;
}

MatrixXf whiten_coordinates_eigen(
    const MatrixXf& x,
    const MatrixXf& whitening_w
) {
    if (x.cols() != whitening_w.rows()) {
        return MatrixXf();
    }
    return x * whitening_w;
}

Status whiten_coordinates_batch(
    const f32* q,
    const f32* k,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* q_whitened,
    f32* k_whitened
) {
    Status status = whiten_coordinates(q, whitening_w, n, d, q_whitened);
    if (status != Status::OK) {
        return status;
    }
    return whiten_coordinates(k, whitening_w, n, d, k_whitened);
}

bool verify_whitening_isometry(
    const MatrixXf& whitening_w,
    const MatrixXf& metric_m,
    size_t d,
    size_t num_samples,
    f32 max_relative_error
) {
    if (d == 0 || num_samples == 0 || whitening_w.rows() != static_cast<Eigen::Index>(d) ||
        whitening_w.cols() != static_cast<Eigen::Index>(d) ||
        metric_m.rows() != static_cast<Eigen::Index>(d) ||
        metric_m.cols() != static_cast<Eigen::Index>(d) ||
        !std::isfinite(max_relative_error) || max_relative_error < 0.0f) {
        return false;
    }
    if (!whitening_w.allFinite() || !metric_m.allFinite()) {
        return false;
    }

    for (size_t sample = 0; sample < num_samples; ++sample) {
        VectorXf x(static_cast<Eigen::Index>(d));
        for (size_t i = 0; i < d; ++i) {
            x(static_cast<Eigen::Index>(i)) = static_cast<f32>(
                std::sin(static_cast<f64>(sample) * 1000.0 + static_cast<f64>(i) * 0.5) * 0.5 + 0.5);
        }
        const f32 norm = x.norm();
        if (!(norm > 0.0f) || !std::isfinite(norm)) {
            return false;
        }
        x /= norm;

        const VectorXf wx = whitening_w * x;
        const f32 wx_norm_sq = wx.squaredNorm();
        const f32 x_m_x = (x.transpose() * metric_m * x)(0, 0);
        if (!std::isfinite(wx_norm_sq) || !std::isfinite(x_m_x)) {
            return false;
        }
        const f32 diff = std::abs(wx_norm_sq - x_m_x);
        const f32 relative_error = x_m_x > 1e-10f ? diff / x_m_x : diff;
        if (relative_error > max_relative_error) {
            return false;
        }
    }
    return true;
}

} // namespace smao
