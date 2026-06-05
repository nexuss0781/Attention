/**
 * @file numerical_guards.cpp
 * @brief Implementation of numerical stability guards
 *
 * Implements Contract 3.1, 3.2, and 3.3:
 * - Exponent range enforcement
 * - Condition number bounds
 * - NaN/Inf propagation prevention
 */

#include "smao_phase1/core/numerical_guards.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace smao {

Status validate_no_nan_inf(const f32* data, size_t n) {
    if (data == nullptr) {
        return Status::InvalidInput;
    }
    
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i])) {
            return Status::NaNInput;
        }
        if (std::isinf(data[i])) {
            return Status::NaNInput;  // Treat Inf as NaN for consistency
        }
    }
    
    return Status::OK;
}

f32 safe_exp(f32 arg, f32 use_log_space_threshold) {
    // Direct computation for safe range
    if (arg >= EXP_ARG_MIN_F32 && arg <= EXP_ARG_MAX_F32) {
        return std::exp(arg);
    }
    
    // Use log-space computation for extreme values
    if (arg > use_log_space_threshold) {
        // For very large values, compute in log-space
        f32 log_result = arg;  // log(exp(arg)) = arg
        // Clip to representable range
        if (log_result > 88.0f) {
            return std::numeric_limits<f32>::infinity();
        }
        return std::exp(log_result);
    }
    
    if (arg < -use_log_space_threshold) {
        // For very negative values, return 0 (underflow)
        return 0.0f;
    }
    
    // Default case
    return std::exp(arg);
}

f32 log_sum_exp(f32 a, f32 b) {
    // log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    if (a > b) {
        return a + std::log1p(std::exp(b - a));
    } else {
        return b + std::log1p(std::exp(a - b));
    }
}

f32 enforce_condition_number_bound(
    f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0 || kappa_max <= 1.0f) {
        return 0.0f;
    }
    
    // Find min and max eigenvalues
    f32 lambda_min = eigenvalues[0];
    f32 lambda_max = eigenvalues[0];
    
    for (size_t i = 1; i < d; ++i) {
        if (eigenvalues[i] < lambda_min) lambda_min = eigenvalues[i];
        if (eigenvalues[i] > lambda_max) lambda_max = eigenvalues[i];
    }
    
    // Enforce minimum eigenvalue threshold
    f32 epsilon = LAMBDA_MIN_THRESHOLD;
    if (lambda_min < epsilon) {
        f32 shift = epsilon - lambda_min;
        for (size_t i = 0; i < d; ++i) {
            eigenvalues[i] += shift;
        }
        lambda_min = epsilon;
        lambda_max += shift;
    }
    
    // Compute current condition number
    f32 kappa = lambda_max / lambda_min;
    
    // If condition number exceeds bound, project eigenvalues
    if (kappa > kappa_max) {
        // Compute target minimum eigenvalue
        f32 lambda_min_target = lambda_max / kappa_max;
        
        // Clip eigenvalues to [lambda_min_target, lambda_max]
        for (size_t i = 0; i < d; ++i) {
            if (eigenvalues[i] < lambda_min_target) {
                eigenvalues[i] = lambda_min_target;
            }
            // Also cap at lambda_max (shouldn't be necessary but for safety)
            if (eigenvalues[i] > lambda_max) {
                eigenvalues[i] = lambda_max;
            }
        }
        
        // Update lambda_min and kappa
        lambda_min = lambda_min_target;
        kappa = kappa_max;
    }
    
    return kappa;
}

bool validate_condition_number(
    const f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0) {
        return false;
    }
    
    f32 lambda_min = eigenvalues[0];
    f32 lambda_max = eigenvalues[0];
    
    for (size_t i = 1; i < d; ++i) {
        if (eigenvalues[i] < lambda_min) lambda_min = eigenvalues[i];
        if (eigenvalues[i] > lambda_max) lambda_max = eigenvalues[i];
    }
    
    if (lambda_min <= 0.0f) {
        return false;
    }
    
    f32 kappa = lambda_max / lambda_min;
    return kappa <= kappa_max;
}

int32_t ulp_difference(f32 a, f32 b) {
    // Handle special cases
    if (std::isnan(a) || std::isnan(b)) {
        return std::numeric_limits<int32_t>::max();
    }
    if (std::isinf(a) && std::isinf(b)) {
        return (a == b) ? 0 : std::numeric_limits<int32_t>::max();
    }
    if (a == b) {
        return 0;
    }
    
    // Reinterpret as integers
    union FloatInt {
        f32 f;
        int32_t i;
    };
    
    FloatInt fa, fb;
    fa.f = a;
    fb.f = b;
    
    // Handle sign bit for proper ordering
    if (fa.i < 0) fa.i = 0x80000000 - fa.i;
    if (fb.i < 0) fb.i = 0x80000000 - fb.i;
    
    return std::abs(fa.i - fb.i);
}

bool approx_equal_ulp(f32 a, f32 b, int32_t max_ulp) {
    return ulp_difference(a, b) <= max_ulp;
}

bool approx_equal_rel(f32 a, f32 b, f32 rel_tol) {
    if (a == b) return true;
    
    f32 abs_a = std::abs(a);
    f32 abs_b = std::abs(b);
    f32 diff = std::abs(a - b);
    
    if (abs_a == 0.0f || abs_b == 0.0f) {
        // One of them is zero, use absolute tolerance
        return diff < rel_tol;
    }
    
    return diff / std::max(abs_a, abs_b) < rel_tol;
}

} // namespace smao
