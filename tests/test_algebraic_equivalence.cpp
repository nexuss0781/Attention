/**
 * @file test_algebraic_equivalence.cpp
 * @brief Comprehensive algebraic equivalence tests
 *
 * Test 1.1: Algebraic Equivalence (The Core Invariant)
 * Validates the exact softmax-to-Gaussian decomposition.
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/phase1_forward.h"
#include "smao_phase1/core/exact_decomposition.h"
#include <random>
#include <cmath>

namespace smao {
namespace {

// Helper function to compute standard softmax attention score
f32 compute_standard_attention_score(
    const f32* q,
    const f32* k,
    size_t d,
    f32 sigma_sq
) {
    f32 dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dot += q[i] * k[i];
    }
    return std::exp(dot / sigma_sq);
}

// Helper function to compute Gaussian decomposition score
f32 compute_gaussian_score(
    const f32* q,
    const f32* k,
    size_t d,
    f32 sq_norm_q,
    f32 sq_norm_k,
    f32 sigma_sq
) {
    f32 denom = 2.0f * sigma_sq;
    f32 a = std::exp(sq_norm_q / denom);
    f32 w = std::exp(sq_norm_k / denom);
    
    f32 dist_sq = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        f32 diff = q[i] - k[i];
        dist_sq += diff * diff;
    }
    
    f32 gaussian = std::exp(-dist_sq / denom);
    
    return a * w * gaussian;
}

// Test 1.1: Algebraic Equivalence (Core Invariant)
TEST(AlgebraicEquivalenceTest, CoreInvariant) {
    constexpr size_t n = 512;
    constexpr size_t d = 64;
    constexpr size_t d_k = d;
    constexpr f32 sigma_sq = std::sqrt(static_cast<f32>(d_k));
    
    // Generate random Q, K from N(0, 1/sqrt(d))
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Test for a subset of pairs
    constexpr size_t num_test_pairs = 1000;
    f32 max_rel_error = 0.0f;
    
    std::uniform_int_distribution<size_t> idx_dist(0, n - 1);
    
    for (size_t test = 0; test < num_test_pairs; ++test) {
        size_t i = idx_dist(gen);
        size_t j = idx_dist(gen);
        
        const f32* q_i = &q[i * d];
        const f32* k_j = &k[j * d];
        
        // Compute standard attention score
        f32 standard_score = compute_standard_attention_score(q_i, k_j, d, sigma_sq);
        
        // Compute squared norms
        f32 sq_norm_q = 0.0f;
        f32 sq_norm_k = 0.0f;
        for (size_t dim = 0; dim < d; ++dim) {
            sq_norm_q += q_i[dim] * q_i[dim];
            sq_norm_k += k_j[dim] * k_j[dim];
        }
        
        // Compute Gaussian decomposition score
        f32 gaussian_score = compute_gaussian_score(q_i, k_j, d, sq_norm_q, sq_norm_k, sigma_sq);
        
        // Compute relative error
        f32 rel_error = std::abs(standard_score - gaussian_score);
        if (standard_score > 1e-10f) {
            rel_error /= standard_score;
        }
        
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    // Pass criterion: relative error < 10^-5 for f32
    EXPECT_LT(max_rel_error, 1e-5f);
}

// Test with different dimensions
TEST(AlgebraicEquivalenceTest, VariousDimensions) {
    std::vector<std::pair<size_t, size_t>> test_cases = {
        {64, 32},
        {128, 64},
        {256, 128},
        {32, 64}
    };
    
    std::mt19937 gen(42);
    
    for (const auto& [n, d] : test_cases) {
        f32 sigma_sq = std::sqrt(static_cast<f32>(d));
        
        std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
        
        std::vector<f32> q(n * d);
        std::vector<f32> k(n * d);
        
        for (size_t i = 0; i < n * d; ++i) {
            q[i] = dist(gen);
            k[i] = dist(gen);
        }
        
        // Test a few pairs
        f32 max_rel_error = 0.0f;
        for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
            for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
                f32 standard = compute_standard_attention_score(&q[i * d], &k[j * d], d, sigma_sq);
                
                f32 sq_norm_q = 0.0f, sq_norm_k = 0.0f;
                for (size_t dim = 0; dim < d; ++dim) {
                    sq_norm_q += q[i * d + dim] * q[i * d + dim];
                    sq_norm_k += k[j * d + dim] * k[j * d + dim];
                }
                
                f32 gaussian = compute_gaussian_score(&q[i * d], &k[j * d], d, sq_norm_q, sq_norm_k, sigma_sq);
                
                f32 rel_error = std::abs(standard - gaussian);
                if (standard > 1e-10f) rel_error /= standard;
                max_rel_error = std::max(max_rel_error, rel_error);
            }
        }
        
        EXPECT_LT(max_rel_error, 1e-5f) << "Failed for n=" << n << ", d=" << d;
    }
}

} // namespace
} // namespace smao
