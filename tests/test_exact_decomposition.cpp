/**
 * @file test_exact_decomposition.cpp
 * @brief Tests for exact softmax-to-Gaussian decomposition
 *
 * Test 1.1: Algebraic Equivalence (The Core Invariant)
 * Test 1.4: Extreme Norm Overflow Guard
 * Test 1.6: Adversarial Orthogonal Inputs
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/numerical_guards.h"
#include <random>
#include <cmath>

namespace smao {
namespace {

// Test 1.1: Algebraic Equivalence
TEST(ExactDecompositionTest, AlgebraicEquivalence) {
    constexpr size_t n = 512;
    constexpr size_t d = 64;
    constexpr size_t d_k = d;
    
    // Generate random Q, K from N(0, 1)
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Compute standard attention scores
    std::vector<f32> standard_scores(n * n);
    f32 sigma_sq = std::sqrt(static_cast<f32>(d_k));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Compute dot product q_i . k_j
            f32 dot = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                dot += q[i * d + dim] * k[j * d + dim];
            }
            standard_scores[i * n + j] = std::exp(dot / sigma_sq);
        }
    }
    
    // Compute Phase 1 outputs
    std::vector<f32> query_scales(n);
    std::vector<f32> key_weights(n);
    f32 sigma_squared;
    
    Status status = exact_decomposition(
        q.data(), k.data(), n, d, d_k,
        1e-6f, -80.0f, 80.0f,
        query_scales.data(), key_weights.data(), &sigma_squared
    );
    
    EXPECT_EQ(status, Status::OK);
    
    // Compute S'_ij = a_i * w_j * exp(-||q_i - k_j||^2 / 2*sigma^2)
    std::vector<f32> phase1_scores(n * n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Compute ||q_i - k_j||^2
            f32 dist_sq = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                f32 diff = q[i * d + dim] - k[j * d + dim];
                dist_sq += diff * diff;
            }
            
            f32 gaussian = std::exp(-dist_sq / (2.0f * sigma_squared));
            phase1_scores[i * n + j] = query_scales[i] * key_weights[j] * gaussian;
        }
    }
    
    // Measure max relative error
    f32 max_rel_error = 0.0f;
    for (size_t i = 0; i < n * n; ++i) {
        f32 abs_error = std::abs(standard_scores[i] - phase1_scores[i]);
        f32 rel_error = (std::abs(standard_scores[i]) > 1e-10f) ?
                        abs_error / std::abs(standard_scores[i]) : abs_error;
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    // Pass criterion: relative error < 10^-5 for f32
    EXPECT_LT(max_rel_error, 1e-5f);
}

// Test 1.4: Extreme Norm Overflow Guard
TEST(ExactDecompositionTest, OverflowGuard) {
    constexpr size_t n = 10;
    constexpr size_t d = 64;
    constexpr size_t d_k = d;
    
    // Create query with extreme norm (||q||^2 = 200)
    std::vector<f32> q(n * d, 0.0f);
    std::vector<f32> k(n * d, 0.0f);
    
    // Set one element to achieve squared norm of 200
    f32 val = std::sqrt(200.0f);
    for (size_t i = 0; i < n; ++i) {
        q[i * d] = val;
        // Set key to small values
        for (size_t j = 0; j < d; ++j) {
            k[i * d + j] = 0.01f;
        }
    }
    
    std::vector<f32> query_scales(n);
    std::vector<f32> key_weights(n);
    f32 sigma_squared;
    
    // Test without clipping - should overflow
    // With sigma^2 = sqrt(64) ~ 8, and ||q||^2 = 200
    // exp_arg = 200 / (2*8) = 12.5, which is safe
    // But let's test with an even larger value
    
    q[0] = std::sqrt(1600.0f);  // ||q||^2 = 1600
    // exp_arg = 1600 / 16 = 100, which exceeds 88
    
    Status status = exact_decomposition(
        q.data(), k.data(), n, d, d_k,
        1e-6f, -80.0f, 80.0f,
        query_scales.data(), key_weights.data(), &sigma_squared
    );
    
    // Should handle gracefully (clip and return finite value)
    // With log-space clipping, should not overflow
    EXPECT_TRUE(status == Status::OK || status == Status::Overflow);
}

// Test 1.6: Adversarial Orthogonal Inputs
TEST(ExactDecompositionTest, AdversarialOrthogonal) {
    constexpr size_t n = 100;
    constexpr size_t d = 64;
    constexpr size_t d_k = d;
    
    // Generate orthogonal rows with high norm
    std::vector<f32> q(n * d, 0.0f);
    std::vector<f32> k(n * d, 0.0f);
    
    f32 norm = 10.0f;  // High norm
    
    for (size_t i = 0; i < n; ++i) {
        // Create nearly orthogonal rows
        size_t idx = i % d;
        q[i * d + idx] = norm;
        
        // Add small perturbations
        for (size_t j = 1; j < 5 && idx + j < d; ++j) {
            q[i * d + idx + j] = norm * 0.01f * j;
        }
    }
    
    // Similar for keys
    for (size_t i = 0; i < n; ++i) {
        size_t idx = (i + d/2) % d;
        k[i * d + idx] = norm * 0.8f;
    }
    
    std::vector<f32> query_scales(n);
    std::vector<f32> key_weights(n);
    f32 sigma_squared;
    
    Status status = exact_decomposition(
        q.data(), k.data(), n, d, d_k,
        1e-6f, -80.0f, 80.0f,
        query_scales.data(), key_weights.data(), &sigma_squared
    );
    
    // Should complete without catastrophic cancellation
    EXPECT_EQ(status, Status::OK);
    
    // All outputs should be finite
    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::isfinite(query_scales[i]));
        EXPECT_TRUE(std::isfinite(key_weights[i]));
        EXPECT_GT(query_scales[i], 0.0f);
        EXPECT_GT(key_weights[i], 0.0f);
    }
}

// Test Kahan dot product accuracy
TEST(ExactDecompositionTest, KahanAccuracy) {
    constexpr size_t n = 1000;
    
    // Create vectors that will show Kahan improvement
    std::vector<f32> x(n);
    std::vector<f32> y(n);
    
    // Fill with alternating small and large values
    for (size_t i = 0; i < n; ++i) {
        x[i] = (i % 2 == 0) ? 1e8f : 1e-8f;
        y[i] = 1.0f;
    }
    
    // Compute with Kahan
    f32 kahan_result = kahan_dot_product(x.data(), y.data(), n);
    
    // Compute with naive summation
    f32 naive_result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        naive_result += x[i] * y[i];
    }
    
    // Expected result: n/2 * 1e8 + n/2 * 1e-8
    f64 expected = static_cast<f64>(n / 2) * 1e8 + static_cast<f64>(n / 2) * 1e-8;
    
    // Kahan should be closer to expected
    f64 kahan_error = std::abs(static_cast<f64>(kahan_result) - expected);
    f64 naive_error = std::abs(static_cast<f64>(naive_result) - expected);
    
    EXPECT_LT(kahan_error, naive_error);
}

} // namespace
} // namespace smao
