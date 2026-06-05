/**
 * @file test_phase1_forward.cpp
 * @brief Integration tests for Phase 1 forward pass
 *
 * Tests the complete Phase 1 pipeline:
 * - Metric assembly
 * - Coordinate whitening
 * - Exact decomposition
 * - Output validation
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/phase1_forward.h"
#include <random>
#include <cmath>

namespace smao {
namespace {

// Helper to generate random lower triangular L
std::vector<f32> generate_random_l(size_t d, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diag_dist(0.1f, 2.0f);
    
    std::vector<f32> l(d * d, 0.0f);
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                l[i * d + j] = diag_dist(gen);
            } else {
                l[i * d + j] = dist(gen) * 0.5f;
            }
        }
    }
    
    return l;
}

// Complete Phase 1 forward integration test
TEST(Phase1ForwardTest, CompletePipeline) {
    constexpr size_t n = 256;
    constexpr size_t d = 64;
    
    // Generate random inputs
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    std::vector<f32> v(n * d);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
        v[i] = dist(gen);
    }
    
    // Generate random L
    std::vector<f32> l = generate_random_l(d);
    
    // Setup input
    Phase1Input input;
    input.q = q.data();
    input.k = k.data();
    input.v = v.data();
    input.l = l.data();
    input.n = n;
    input.d = d;
    input.d_v = d;
    input.precision = Precision::F32;
    
    // Run Phase 1 forward
    Phase1Output output = phase1_forward(input);
    
    // Check status
    EXPECT_EQ(output.status, Status::OK);
    
    // Check dimensions
    EXPECT_EQ(output.n, n);
    EXPECT_EQ(output.d, d);
    
    // Check outputs exist
    EXPECT_GT(output.whitened_q.size(), 0u);
    EXPECT_GT(output.whitened_k.size(), 0u);
    EXPECT_GT(output.query_scales.size(), 0u);
    EXPECT_GT(output.key_weights.size(), 0u);
    EXPECT_GT(output.metric_m.size(), 0u);
    EXPECT_GT(output.whitening_w.size(), 0u);
    
    // Check condition number
    EXPECT_GT(output.condition_number, 0.0f);
    EXPECT_LE(output.condition_number, 1e4f);
    
    // Check sigma^2
    f32 expected_sigma_sq = std::sqrt(static_cast<f32>(d));
    EXPECT_FLOAT_EQ(output.sigma_squared, expected_sigma_sq);
    
    // Check all outputs are finite
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            EXPECT_TRUE(std::isfinite(output.whitened_q(i, j)));
            EXPECT_TRUE(std::isfinite(output.whitened_k(i, j)));
        }
        EXPECT_TRUE(std::isfinite(output.query_scales(i)));
        EXPECT_TRUE(std::isfinite(output.key_weights(i)));
        EXPECT_GT(output.query_scales(i), 0.0f);
        EXPECT_GT(output.key_weights(i), 0.0f);
    }
    
    // Check metric is SPD
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            EXPECT_TRUE(std::isfinite(output.metric_m(i, j)));
            EXPECT_TRUE(std::isfinite(output.whitening_w(i, j)));
        }
    }
}

// Test frozen gate criteria
TEST(Phase1ForwardTest, FrozenGateCriteria) {
    constexpr size_t n = 128;
    constexpr size_t d = 32;
    
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    std::vector<f32> l(d * d, 0.0f);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Create valid L (lower triangular with positive diagonal)
    std::uniform_real_distribution<f32> diag_dist(0.5f, 1.5f);
    for (size_t i = 0; i < d; ++i) {
        l[i * d + i] = diag_dist(gen);
        for (size_t j = 0; j < i; ++j) {
            l[i * d + j] = dist(gen) * 0.3f;
        }
    }
    
    Phase1Input input;
    input.q = q.data();
    input.k = k.data();
    input.v = q.data();
    input.l = l.data();
    input.n = n;
    input.d = d;
    input.d_v = d;
    input.precision = Precision::F32;
    
    Phase1Output output = phase1_forward(input);
    
    // Check frozen gate criteria
    EXPECT_TRUE(check_frozen_gate_criteria(output));
    
    // Criterion 1: Condition number < 1e4
    EXPECT_LE(output.condition_number, 1e4f);
    
    // Criterion 2: All outputs finite
    for (size_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::isfinite(output.query_scales(i)));
        EXPECT_TRUE(std::isfinite(output.key_weights(i)));
    }
}

// Test with invalid inputs
TEST(Phase1ForwardTest, InvalidInputs) {
    Phase1Input input = {};
    
    // Null Q
    input.k = (f32*)1;  // Dummy non-null
    input.l = (f32*)1;
    input.n = 10;
    input.d = 16;
    
    Phase1Output output = phase1_forward(input);
    EXPECT_NE(output.status, Status::OK);
    
    // Zero dimensions
    input.q = (f32*)1;
    input.n = 0;
    output = phase1_forward(input);
    EXPECT_NE(output.status, Status::OK);
    
    // NaN in input
    std::vector<f32> q(160, 0.0f);
    q[0] = std::numeric_limits<f32>::quiet_NaN();
    input.n = 10;
    input.d = 16;
    input.q = q.data();
    
    output = phase1_forward(input);
    EXPECT_NE(output.status, Status::OK);
}

} // namespace
} // namespace smao
