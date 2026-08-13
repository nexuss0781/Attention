/*
 * @file numerical_guards.cpp
 * @brief Numerical stability guards and validation functions.
 */

#include "smao_phase1/core/numerical_guards.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace smao {

Status validate_no_nan_inf(const f32* data, size_t n) {
    if (data == nullptr && n != 0) return Status::InvalidInput;
    if (n == 0) return Status::OK;
    if (n > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) {
        return Status::InvalidInput;
    }
    const Eigen::Map<const VectorXf> values(data, static_cast<Eigen::Index>(n));
    return values.allFinite() ? Status::OK : Status::NaNInput;
}

f32 safe_exp(f32 arg, f32 use_log_space_threshold) {
    if (!std::isfinite(arg) || !std::isfinite(use_log_space_threshold) ||
        use_log_space_threshold <= 0.0f) {
        return std::numeric_limits<f32>::quiet_NaN();
    }

    constexpr f32 log_max_f32 = 88.72283905206835f;
    if (arg >= log_max_f32 || arg >= use_log_space_threshold) {
        return std::numeric_limits<f32>::max();
    }
    if (arg <= -use_log_space_threshold) {
        return 0.0f;
    }
    return std::exp(arg);
}

f32 log_sum_exp(f32 a, f32 b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    if (a == -std::numeric_limits<f32>::infinity()) {
        return b;
    }
    if (b == -std::numeric_limits<f32>::infinity()) {
        return a;
    }
    const f32 maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

f32 enforce_condition_number_bound(
    f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0 || !std::isfinite(kappa_max) || kappa_max <= 1.0f) {
        return 0.0f;
    }

    f32 lambda_min = std::numeric_limits<f32>::infinity();
    f32 lambda_max = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        if (!std::isfinite(eigenvalues[i])) {
            return 0.0f;
        }
        lambda_min = std::min(lambda_min, eigenvalues[i]);
        lambda_max = std::max(lambda_max, eigenvalues[i]);
    }

    if (!(lambda_max > 0.0f)) {
        return 0.0f;
    }
    if (lambda_min < LAMBDA_MIN_THRESHOLD) {
        const f32 shift = LAMBDA_MIN_THRESHOLD - lambda_min;
        for (size_t i = 0; i < d; ++i) {
            eigenvalues[i] += shift;
        }
        lambda_min += shift;
        lambda_max += shift;
    }

    const f32 target_min = std::max(LAMBDA_MIN_THRESHOLD, lambda_max / kappa_max);
    for (size_t i = 0; i < d; ++i) {
        eigenvalues[i] = std::max(eigenvalues[i], target_min);
    }

    lambda_min = eigenvalues[0];
    lambda_max = eigenvalues[0];
    for (size_t i = 1; i < d; ++i) {
        lambda_min = std::min(lambda_min, eigenvalues[i]);
        lambda_max = std::max(lambda_max, eigenvalues[i]);
    }
    return lambda_max / lambda_min;
}

bool validate_condition_number(
    const f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0 || !std::isfinite(kappa_max) || kappa_max <= 1.0f) {
        return false;
    }

    f32 lambda_min = std::numeric_limits<f32>::infinity();
    f32 lambda_max = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        if (!std::isfinite(eigenvalues[i]) || !(eigenvalues[i] > 0.0f)) {
            return false;
        }
        lambda_min = std::min(lambda_min, eigenvalues[i]);
        lambda_max = std::max(lambda_max, eigenvalues[i]);
    }
    return lambda_max / lambda_min <= kappa_max;
}

int32_t ulp_difference(f32 a, f32 b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return std::numeric_limits<int32_t>::max();
    }
    if (a == b) {
        return 0;
    }

    auto ordered_bits = [](f32 value) -> uint32_t {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(value));
        if ((bits & 0x80000000u) != 0u) {
            return ~bits + 1u;
        }
        return bits | 0x80000000u;
    };

    const uint32_t ia = ordered_bits(a);
    const uint32_t ib = ordered_bits(b);
    const uint64_t difference = ia > ib ? static_cast<uint64_t>(ia - ib) : static_cast<uint64_t>(ib - ia);
    return difference > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())
        ? std::numeric_limits<int32_t>::max()
        : static_cast<int32_t>(difference);
}

bool approx_equal_ulp(f32 a, f32 b, int32_t max_ulp) {
    return max_ulp >= 0 && ulp_difference(a, b) <= max_ulp;
}

bool approx_equal_rel(f32 a, f32 b, f32 rel_tol) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(rel_tol) || rel_tol < 0.0f) {
        return false;
    }
    if (a == b) {
        return true;
    }
    const f32 scale = std::max(std::abs(a), std::abs(b));
    if (scale == 0.0f) {
        return true;
    }
    return std::abs(a - b) <= rel_tol * scale;
}

} // namespace smao
