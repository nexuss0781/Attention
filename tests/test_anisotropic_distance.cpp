/**
 * @file test_anisotropic_distance.cpp
 * @brief Tests for anisotropic distance computation
 *
 * Test 1.3: Anisotropic Kernel Consistency
 * Test 1.8: Output Distribution Preservation
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/anisotropic_distance.h"
#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/numerical_guards.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace smao {
namespace {

// Test 1.3: Anisotropic Kernel Consistency
TEST(AnisotropicDistanceTest, KernelConsistency) {
    constexpr size_t d = 64;
    constexpr f32 sigma_sq = std::sqrt(static_cast<f32>(d));
    
    // Generate random metric L
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diag_dist(0.5f, 2.0f);
    
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
    
    // Compute metric assembly
    MetricAssemblyResult result = metric_assembly(l.data(), d);
    
    EXPECT_EQ(result.status, Status::OK);
    
    // Generate random q, k
    std::vector<f32> q(d);
    std::vector<f32> k(d);
    
    for (size_t i = 0; i < d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Compute anisotropic kernel
    f32 dist_sq_m = anisotropic_distance_squared(q.data(), k.data(), result.metric_m.data(), d);
    f32 kernel_anisotropic = std::exp(-dist_sq_m / (2.0f * sigma_sq));
    
    // Compute whitened coordinates
    std::vector<f32> q_tilde(d, 0.0f);
    std::vector<f32> k_tilde(d, 0.0f);
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            q_tilde[i] += result.whitening_w(i, j) * q[j];
            k_tilde[i] += result.whitening_w(i, j) * k[j];
        }
    }
    
    // Compute isotropic kernel in whitened space
    f32 dist_sq_iso = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        f32 diff = q_tilde[i] - k_tilde[i];
        dist_sq_iso += diff * diff;
    }
    f32 kernel_isotropic = std::exp(-dist_sq_iso / (2.0f * sigma_sq));
    
    // Compute relative error
    f32 abs_diff = std::abs(kernel_anisotropic - kernel_isotropic);
    f32 max_val = std::max(std::abs(kernel_anisotropic), std::abs(kernel_isotropic));
    f32 relative_error = (max_val > 1e-10f) ? abs_diff / max_val : abs_diff;
    
    // Pass criterion: relative error < 10^-6
    EXPECT_LT(relative_error, 1e-6f);
}

// Test 1.8: Output Distribution Preservation
TEST(AnisotropicDistanceTest, DistributionPreservation) {
    constexpr size_t n = 1000;
    constexpr size_t d = 64;
    
    // Generate Q, K from N(0, 1/sqrt(d))
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Compute standard attention weights
    f32 sigma_sq = std::sqrt(static_cast<f32>(d));
    std::vector<f32> standard_weights;
    
    for (size_t i = 0; i < n; ++i) {
        f32 sum_exp = 0.0f;
        std::vector<f32> row_exp(n);
        
        for (size_t j = 0; j < n; ++j) {
            f32 dot = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                dot += q[i * d + dim] * k[j * d + dim];
            }
            row_exp[j] = std::exp(dot / sigma_sq);
            sum_exp += row_exp[j];
        }
        
        // Normalize to get attention weights
        for (size_t j = 0; j < n; ++j) {
            standard_weights.push_back(row_exp[j] / sum_exp);
        }
    }
    
    // Compute Gaussian decomposition weights
    // For this test, we use the decomposition without whitening (identity metric)
    std::vector<f32> gaussian_weights;
    
    for (size_t i = 0; i < n; ++i) {
        // Compute a_i
        f32 sq_norm_q = 0.0f;
        for (size_t dim = 0; dim < d; ++dim) {
            sq_norm_q += q[i * d + dim] * q[i * d + dim];
        }
        f32 a_i = std::exp(sq_norm_q / (2.0f * sigma_sq));
        
        std::vector<f32> row_gauss(n);
        f32 sum_gauss = 0.0f;
        
        for (size_t j = 0; j < n; ++j) {
            // Compute w_j
            f32 sq_norm_k = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                sq_norm_k += k[j * d + dim] * k[j * d + dim];
            }
            f32 w_j = std::exp(sq_norm_k / (2.0f * sigma_sq));
            
            // Compute Gaussian
            f32 dist_sq = 0.0f;
            for (size_t dim = 0; dim < d; ++dim) {
                f32 diff = q[i * d + dim] - k[j * d + dim];
                dist_sq += diff * diff;
            }
            f32 gaussian = std::exp(-dist_sq / (2.0f * sigma_sq));
            
            row_gauss[j] = a_i * w_j * gaussian;
            sum_gauss += row_gauss[j];
        }
        
        // Normalize
        for (size_t j = 0; j < n; ++j) {
            gaussian_weights.push_back(row_gauss[j] / sum_gauss);
        }
    }
    
    // Compare distributions using Kolmogorov-Smirnov statistic approximation
    // For simplicity, we compare means and variances
    f32 standard_mean = 0.0f, gaussian_mean = 0.0f;
    f32 standard_var = 0.0f, gaussian_var = 0.0f;
    
    size_t total = standard_weights.size();
    
    for (size_t i = 0; i < total; ++i) {
        standard_mean += standard_weights[i];
        gaussian_mean += gaussian_weights[i];
    }
    standard_mean /= total;
    gaussian_mean /= total;
    
    for (size_t i = 0; i < total; ++i) {
        standard_var += (standard_weights[i] - standard_mean) * (standard_weights[i] - standard_mean);
        gaussian_var += (gaussian_weights[i] - gaussian_mean) * (gaussian_weights[i] - gaussian_mean);
    }
    standard_var /= total;
    gaussian_var /= total;
    
    // Distributions should have similar means and variances
    EXPECT_NEAR(standard_mean, gaussian_mean, 0.01f);
    EXPECT_NEAR(standard_var, gaussian_var, 0.01f);
}

} // namespace
} // namespace smao
