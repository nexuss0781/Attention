/**
 * @file test_numerical_guards.cpp
 * @brief Tests for numerical stability guards
 *
 * Test 1.4: Extreme Norm Overflow Guard
 * Contract 3.1: Exponent range enforcement
 * Contract 3.2: Condition number bounds
 * Contract 3.3: NaN propagation prevention
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/numerical_guards.h"
#include <cmath>
#include <limits>

namespace smao {
namespace {

// Test NaN detection
TEST(NumericalGuardsTest, NaNDetection) {
    f32 nan_val = std::numeric_limits<f32>::quiet_NaN();
    f32 inf_val = std::numeric_limits<f32>::infinity();
    f32 normal_val = 1.0f;
    f32 zero_val = 0.0f;
    
    EXPECT_TRUE(is_nan(nan_val));
    EXPECT_FALSE(is_nan(inf_val));
    EXPECT_FALSE(is_nan(normal_val));
    EXPECT_FALSE(is_nan(zero_val));
    
    EXPECT_TRUE(is_finite(normal_val));
    EXPECT_TRUE(is_finite(zero_val));
    EXPECT_FALSE(is_finite(inf_val));
    EXPECT_FALSE(is_finite(-inf_val));
    EXPECT_FALSE(is_finite(nan_val));
}

// Test validate_no_nan_inf
TEST(NumericalGuardsTest, ValidateNoNaNInf) {
    std::vector<f32> valid_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_EQ(validate_no_nan_inf(valid_data.data(), valid_data.size()), Status::OK);
    
    std::vector<f32> nan_data = {1.0f, 2.0f, std::numeric_limits<f32>::quiet_NaN(), 4.0f};
    EXPECT_EQ(validate_no_nan_inf(nan_data.data(), nan_data.size()), Status::NaNInput);
    
    std::vector<f32> inf_data = {1.0f, std::numeric_limits<f32>::infinity(), 3.0f};
    EXPECT_EQ(validate_no_nan_inf(inf_data.data(), inf_data.size()), Status::NaNInput);
    
    EXPECT_EQ(validate_no_nan_inf(nullptr, 10), Status::InvalidInput);
    EXPECT_EQ(validate_no_nan_inf(valid_data.data(), 0), Status::OK);
}

// Test exponent clipping
TEST(NumericalGuardsTest, ExponentClipping) {
    // Test normal range
    EXPECT_EQ(clip_exponent_arg(0.0f), 0.0f);
    EXPECT_EQ(clip_exponent_arg(10.0f), 10.0f);
    EXPECT_EQ(clip_exponent_arg(-10.0f), -10.0f);
    
    // Test clipping
    EXPECT_EQ(clip_exponent_arg(100.0f), 80.0f);
    EXPECT_EQ(clip_exponent_arg(-100.0f), -80.0f);
    
    // Test custom bounds
    EXPECT_EQ(clip_exponent_arg(50.0f, -40.0f, 40.0f), 40.0f);
    EXPECT_EQ(clip_exponent_arg(-50.0f, -40.0f, 40.0f), -40.0f);
}

// Test safe_exp
TEST(NumericalGuardsTest, SafeExp) {
    // Normal range
    EXPECT_NEAR(safe_exp(0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(safe_exp(1.0f), std::exp(1.0f), 1e-5f);
    EXPECT_NEAR(safe_exp(-1.0f), std::exp(-1.0f), 1e-5f);
    
    // Large positive (should not overflow)
    f32 large_exp = safe_exp(100.0f);
    EXPECT_TRUE(std::isfinite(large_exp));
    
    // Large negative (should underflow to 0)
    f32 small_exp = safe_exp(-100.0f);
    EXPECT_EQ(small_exp, 0.0f);
}

// Test log_sum_exp
TEST(NumericalGuardsTest, LogSumExp) {
    // Simple cases
    EXPECT_NEAR(log_sum_exp(0.0f, 0.0f), std::log(2.0f), 1e-5f);
    EXPECT_NEAR(log_sum_exp(1.0f, 0.0f), std::log(1.0f + std::exp(1.0f)), 1e-5f);
    
    // Large values (should not overflow)
    f32 result = log_sum_exp(100.0f, 100.0f);
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_NEAR(result, 100.0f + std::log(2.0f), 1e-4f);
    
    // Mixed large and small
    result = log_sum_exp(100.0f, 0.0f);
    EXPECT_TRUE(std::isfinite(result));
}

// Test condition number enforcement
TEST(NumericalGuardsTest, ConditionNumberEnforcement) {
    constexpr size_t d = 16;
    constexpr f32 kappa_max = 100.0f;
    
    // Test case 1: Well-conditioned eigenvalues (should not change)
    std::vector<f32> eigenvalues1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                      9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    std::vector<f32> eigenvalues1_copy = eigenvalues1;
    
    f32 kappa1 = enforce_condition_number_bound(eigenvalues1.data(), d, kappa_max);
    
    EXPECT_LE(kappa1, kappa_max);
    // Should not change (already well-conditioned)
    for (size_t i = 0; i < d; ++i) {
        EXPECT_FLOAT_EQ(eigenvalues1[i], eigenvalues1_copy[i]);
    }
    
    // Test case 2: Ill-conditioned eigenvalues (needs projection)
    std::vector<f32> eigenvalues2(d);
    for (size_t i = 0; i < d; ++i) {
        eigenvalues2[i] = 1.0f + static_cast<f32>(i) * 100.0f;  // Range: 1 to 1501
    }
    
    f32 kappa2 = enforce_condition_number_bound(eigenvalues2.data(), d, kappa_max);
    
    EXPECT_LE(kappa2, kappa_max);
    
    // Verify eigenvalues are sorted and clipped
    for (size_t i = 1; i < d; ++i) {
        EXPECT_LE(eigenvalues2[i-1], eigenvalues2[i]);
    }
    
    // Test case 3: Very small eigenvalues (should be lifted)
    std::vector<f32> eigenvalues3(d);
    for (size_t i = 0; i < d; ++i) {
        eigenvalues3[i] = static_cast<f32>(i + 1) * 1e-7f;  // Very small
    }
    
    f32 kappa3 = enforce_condition_number_bound(eigenvalues3.data(), d, kappa_max);
    
    EXPECT_LE(kappa3, kappa_max);
    
    // All eigenvalues should be >= 1e-6 (LAMBDA_MIN_THRESHOLD)
    for (size_t i = 0; i < d; ++i) {
        EXPECT_GE(eigenvalues3[i], LAMBDA_MIN_THRESHOLD);
    }
}

// Test ULP difference computation
TEST(NumericalGuardsTest, ULPDifference) {
    // Same value: 0 ULP
    EXPECT_EQ(ulp_difference(1.0f, 1.0f), 0);
    
    // Close values
    f32 a = 1.0f;
    f32 b = std::nextafter(a, 2.0f);
    EXPECT_EQ(ulp_difference(a, b), 1);
    
    // More distant values
    f32 c = std::nextafter(b, 2.0f);
    EXPECT_EQ(ulp_difference(a, c), 2);
    
    // Special values
    f32 inf = std::numeric_limits<f32>::infinity();
    f32 nan = std::numeric_limits<f32>::quiet_NaN();
    
    EXPECT_GT(ulp_difference(inf, inf), 1000000);  // Should be large
    EXPECT_GT(ulp_difference(nan, 1.0f), 1000000);  // Should be large
}

// Test approximate equality
TEST(NumericalGuardsTest, ApproximateEquality) {
    // Exactly equal
    EXPECT_TRUE(approx_equal_ulp(1.0f, 1.0f, 10));
    EXPECT_TRUE(approx_equal_rel(1.0f, 1.0f, 1e-5f));
    
    // Close values
    f32 a = 1.0f;
    f32 b = std::nextafter(a, 2.0f);
    EXPECT_TRUE(approx_equal_ulp(a, b, 10));
    EXPECT_TRUE(approx_equal_rel(a, b, 1e-5f));
    
    // Different values
    f32 c = 2.0f;
    EXPECT_FALSE(approx_equal_ulp(a, c, 10));
    EXPECT_FALSE(approx_equal_rel(a, c, 1e-5f));
}

} // namespace
} // namespace smao
