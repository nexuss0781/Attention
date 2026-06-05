#pragma once

#include "phase1_types.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace smao::phase1 {

// ============================================================================
// Input Validation
// ============================================================================

/**
 * Check if a value is NaN or Inf
 */
inline bool is_valid(f32 x) {
    return std::isfinite(x);
}

/**
 * Validate a matrix for NaN/Inf
 */
inline Status validate_matrix(const MatrixXf& M, const char* name = "matrix") {
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (!is_valid(M(i, j))) {
                if (std::isnan(M(i, j))) {
                    return Status::INPUT_NAN;
                } else {
                    return Status::INPUT_INF;
                }
            }
        }
    }
    return Status::OK;
}

/**
 * Validate a vector for NaN/Inf
 */
inline Status validate_vector(const VectorXf& v, const char* name = "vector") {
    for (int i = 0; i < v.size(); ++i) {
        if (!is_valid(v(i))) {
            if (std::isnan(v(i))) {
                return Status::INPUT_NAN;
            } else {
                return Status::INPUT_INF;
            }
        }
    }
    return Status::OK;
}

// ============================================================================
// Kahan-Compensated Dot Product
// ============================================================================

/**
 * Compute dot product with Kahan summation for increased precision.
 * This reduces floating-point error accumulation in inner products.
 */
inline f32 kahan_dot_product(const VectorXf& a, const VectorXf& b) {
    f32 sum = 0.0f;
    f32 c = 0.0f;  // Compensation
    
    for (int i = 0; i < a.size(); ++i) {
        f32 y = a(i) * b(i) - c;
        f32 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

/**
 * Compute squared norm with Kahan compensation
 */
inline f32 kahan_squared_norm(const VectorXf& x) {
    f32 sum = 0.0f;
    f32 c = 0.0f;
    
    for (int i = 0; i < x.size(); ++i) {
        f32 y = x(i) * x(i) - c;
        f32 t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}

// ============================================================================
// Exponent Range Clipping
// ============================================================================

/**
 * Clip exponent argument to safe range to prevent overflow/underflow
 */
inline f32 safe_exp_arg(f32 x) {
    return std::clamp(x, constants::EXPONENT_MIN_SAFE, constants::EXPONENT_MAX_SAFE);
}

/**
 * Compute exp with automatic range clipping
 */
inline f32 safe_expf(f32 x) {
    x = safe_exp_arg(x);
    return std::exp(x);
}

// ============================================================================
// SPD Matrix Validation
// ============================================================================

/**
 * Check if eigenvalues satisfy SPD constraints
 */
inline Status validate_spd_eigenvalues(const VectorXf& eigenvalues) {
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < constants::MIN_EIGENVALUE) {
            return Status::METRIC_NOT_SPD;
        }
    }
    return Status::OK;
}

/**
 * Compute condition number from eigenvalues
 */
inline f32 compute_condition_number(const VectorXf& eigenvalues) {
    if (eigenvalues.size() == 0) return 1.0f;
    
    f32 lambda_max = eigenvalues.maxCoeff();
    f32 lambda_min = eigenvalues.minCoeff();
    
    if (lambda_min < 1e-10f) lambda_min = 1e-10f;  // Guard against division by near-zero
    
    return lambda_max / lambda_min;
}

/**
 * Clip eigenvalues to enforce condition number constraint
 */
inline void enforce_condition_number(VectorXf& eigenvalues, f32 max_kappa) {
    if (eigenvalues.size() == 0) return;
    
    f32 lambda_max = eigenvalues.maxCoeff();
    f32 lambda_min_target = lambda_max / max_kappa;
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues(i) = std::max(eigenvalues(i), lambda_min_target);
    }
}

// ============================================================================
// Relative Error Computation
// ============================================================================

/**
 * Compute relative error safely, handling near-zero reference values
 */
inline f32 relative_error(f32 computed, f32 reference) {
    if (std::abs(reference) < 1e-10f) {
        // For very small reference values, use absolute error
        return std::abs(computed - reference);
    }
    return std::abs(computed - reference) / std::abs(reference);
}

/**
 * Compute maximum relative error between two vectors
 */
inline f32 max_relative_error(const VectorXf& computed, const VectorXf& reference) {
    f32 max_err = 0.0f;
    for (int i = 0; i < computed.size(); ++i) {
        max_err = std::max(max_err, relative_error(computed(i), reference(i)));
    }
    return max_err;
}

/**
 * Compute maximum relative error between two matrices (Frobenius sense)
 */
inline f32 max_relative_error(const MatrixXf& computed, const MatrixXf& reference) {
    f32 max_err = 0.0f;
    for (int i = 0; i < computed.rows(); ++i) {
        for (int j = 0; j < computed.cols(); ++j) {
            max_err = std::max(max_err, relative_error(computed(i, j), reference(i, j)));
        }
    }
    return max_err;
}

}  // namespace smao::phase1
