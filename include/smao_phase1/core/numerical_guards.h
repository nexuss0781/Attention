/**
 * @file numerical_guards.h
 * @brief Numerical stability guards and validation functions
 *
 * Implements Contract 3.1, 3.2, and 3.3:
 * - Exponent range enforcement
 * - Condition number bounds
 * - NaN/Inf propagation prevention
 */

#ifndef SMAO_NUMERICAL_GUARDS_H
#define SMAO_NUMERICAL_GUARDS_H

#include "types.h"
#include <cmath>
#include <limits>

namespace smao {

// Constants for numerical stability
constexpr f32 EPSILON_F32 = 1e-6f;
constexpr f32 EXP_ARG_MIN_F32 = -88.0f;  // exp(-88) ~ 1.2e-38
constexpr f32 EXP_ARG_MAX_F32 = 88.0f;   // exp(88) ~ 1.6e38
constexpr f32 EXP_ARG_SAFE_MIN_F32 = -80.0f;  // Safe clipping bounds
constexpr f32 EXP_ARG_SAFE_MAX_F32 = 80.0f;
constexpr f32 CONDITION_NUMBER_MAX_DEFAULT = 1e4f;
constexpr f32 LAMBDA_MIN_THRESHOLD = 1e-6f;

/**
 * @brief Check if value is finite (not NaN or Inf)
 *
 * @param x Value to check
 * @return true if finite
 */
inline bool is_finite(f32 x) {
    return std::isfinite(x);
}

/**
 * @brief Check if value is NaN
 *
 * @param x Value to check
 * @return true if NaN
 */
inline bool is_nan(f32 x) {
    return std::isnan(x);
}

/**
 * @brief Validate that array contains no NaN or Inf values
 *
 * Implements Contract 3.3: NaN propagation prevention
 *
 * @param data Array to validate
 * @param n Number of elements
 * @return Status::OK if valid, Status::NaNInput if NaN/Inf found
 */
Status validate_no_nan_inf(const f32* data, size_t n);

/**
 * @brief Clip exponent argument to safe range
 *
 * Implements Contract 3.1: Exponent range enforcement
 * Prevents underflow/overflow in exp() computation.
 *
 * @param arg Raw exponent argument
 * @param min_val Minimum allowed value (default: -80.0)
 * @param max_val Maximum allowed value (default: 80.0)
 * @return Clipped argument
 */
inline f32 clip_exponent_arg(f32 arg, f32 min_val = EXP_ARG_SAFE_MIN_F32, f32 max_val = EXP_ARG_SAFE_MAX_F32) {
    if (arg < min_val) return min_val;
    if (arg > max_val) return max_val;
    return arg;
}

/**
 * @brief Compute exp with log-space clipping
 *
 * For large arguments, computes in log-space to prevent overflow.
 *
 * @param arg Exponent argument
 * @param use_log_space_threshold Threshold for switching to log-space (default: 80.0)
 * @return exp(arg) with overflow protection
 */
f32 safe_exp(f32 arg, f32 use_log_space_threshold = EXP_ARG_SAFE_MAX_F32);

/**
 * @brief Compute log-sum-exp in numerically stable way
 *
 * Computes log(exp(a) + exp(b)) without overflow.
 *
 * @param a First log-value
 * @param b Second log-value
 * @return log(exp(a) + exp(b))
 */
f32 log_sum_exp(f32 a, f32 b);

/**
 * @brief Enforce condition number bound via spectral projection
 *
 * Implements Contract 3.2: Metric condition number enforcement
 * Clips eigenvalues to ensure kappa <= kappa_max.
 *
 * @param eigenvalues Input/output eigenvalues (size d, modified in place)
 * @param d Dimension
 * @param kappa_max Maximum allowed condition number
 * @return Actual condition number after projection
 */
f32 enforce_condition_number_bound(
    f32* eigenvalues,
    size_t d,
    f32 kappa_max = CONDITION_NUMBER_MAX_DEFAULT
);

/**
 * @brief Validate condition number of SPD matrix
 *
 * @param eigenvalues Eigenvalues of matrix (size d)
 * @param d Dimension
 * @param kappa_max Maximum allowed condition number
 * @return true if condition number is within bound
 */
bool validate_condition_number(
    const f32* eigenvalues,
    size_t d,
    f32 kappa_max = CONDITION_NUMBER_MAX_DEFAULT
);

/**
 * @brief Compute ULP (units in last place) difference between two floats
 *
 * Used for numerical accuracy testing.
 *
 * @param a First value
 * @param b Second value
 * @return ULP difference
 */
int32_t ulp_difference(f32 a, f32 b);

/**
 * @brief Check if two floats are approximately equal within ULP tolerance
 *
 * @param a First value
 * @param b Second value
 * @param max_ulp Maximum allowed ULP difference
 * @return true if approximately equal
 */
bool approx_equal_ulp(f32 a, f32 b, int32_t max_ulp = 10);

/**
 * @brief Check if two floats are approximately equal within relative tolerance
 *
 * @param a First value
 * @param b Second value
 * @param rel_tol Relative tolerance
 * @return true if approximately equal
 */
bool approx_equal_rel(f32 a, f32 b, f32 rel_tol = 1e-5f);

} // namespace smao

#endif // SMAO_NUMERICAL_GUARDS_H
