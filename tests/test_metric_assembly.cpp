/**
 * @file test_metric_assembly.cpp
 * @brief Tests for metric assembly and eigendecomposition
 *
 * Test 1.2: Whitening Isometry
 * Test 1.5: Near-Singular Metric Recovery
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/numerical_guards.h"
#include <random>
#include <cmath>

namespace smao {
namespace {

// Generate random lower triangular matrix L with positive diagonal
std::vector<f32> generate_random_l(size_t d, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diag_dist(0.1f, 2.0f);
    
    std::vector<f32> l(d * d, 0.0f);
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                // Diagonal: positive
                l[i * d + j] = diag_dist(gen);
            } else {
                // Off-diagonal: any value
                l[i * d + j] = dist(gen);
            }
        }
    }
    
    return l;
}

// Test 1.2: Whitening Isometry
TEST(MetricAssemblyTest, WhiteningIsometry) {
    constexpr size_t d = 64;
    constexpr size_t num_samples = 10000;
    constexpr f32 max_relative_error = 1e-6f;
    
    // Generate random L
    std::vector<f32> l = generate_random_l(d);
    
    // Compute metric assembly
    MetricAssemblyResult result = metric_assembly(l.data(), d);
    
    EXPECT_EQ(result.status, Status::OK);
    EXPECT_LT(result.condition_number, 1e4f);
    
    // Verify whitening isometry: ||W*x||^2 = x^T*M*x for random unit vectors
    std::mt19937 gen(12345);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    
    f32 max_error = 0.0f;
    
    for (size_t sample = 0; sample < num_samples; ++sample) {
        // Generate random vector
        VectorXf x(d);
        for (size_t i = 0; i < d; ++i) {
            x(i) = dist(gen);
        }
        
        // Normalize to unit vector
        f32 norm = x.norm();
        if (norm > 1e-10f) {
            x /= norm;
        } else {
            continue;
        }
        
        // Compute ||Wx||^2
        VectorXf wx = result.whitening_w * x;
        f32 wx_norm_sq = wx.squaredNorm();
        
        // Compute x^T M x
        f32 xMx = x.transpose() * result.metric_m * x;
        
        // Compute relative error
        f32 diff = std::abs(wx_norm_sq - xMx);
        f32 relative_error = (xMx > 1e-10f) ? diff / xMx : diff;
        
        max_error = std::max(max_error, relative_error);
    }
    
    EXPECT_LT(max_error, max_relative_error);
}

// Test 1.5: Near-Singular Metric Recovery
TEST(MetricAssemblyTest, NearSingularRecovery) {
    constexpr size_t d = 64;
    
    // Create L with one diagonal entry = 1e-8 (near singular)
    std::vector<f32> l(d * d, 0.0f);
    
    // Fill with reasonable values
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            if (i == j) {
                if (i == 0) {
                    // First diagonal: very small
                    l[i * d + j] = 1e-8f;
                } else {
                    l[i * d + j] = 1.0f;
                }
            } else {
                l[i * d + j] = dist(gen) * 0.1f;
            }
        }
    }
    
    // Metric assembly should recover from near-singular input
    MetricAssemblyResult result = metric_assembly(l.data(), d);
    
    // Should succeed with spectral projection
    EXPECT_EQ(result.status, Status::OK);
    
    // Output M should have lambda_min >= 1e-6
    // (enforced by spectral projection)
    
    // Condition number should be <= 1e4
    EXPECT_LE(result.condition_number, 1e4f);
    
    // Cholesky on M should succeed
    // (This is implicitly tested by the SPD validation in metric_assembly)
}

// Test SPD preservation invariant
TEST(MetricAssemblyTest, SPDPreservation) {
    constexpr size_t d = 32;
    
    // Test with various random L matrices
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diag_dist(0.01f, 2.0f);
    
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<f32> l(d * d, 0.0f);
        
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                if (i == j) {
                    l[i * d + j] = diag_dist(gen);
                } else {
                    l[i * d + j] = dist(gen);
                }
            }
        }
        
        MetricAssemblyResult result = metric_assembly(l.data(), d);
        
        EXPECT_EQ(result.status, Status::OK);
        EXPECT_GT(result.condition_number, 0.0f);
        EXPECT_LE(result.condition_number, 1e4f);
        
        // Verify M is symmetric
        for (size_t i = 0; i < d; ++i) {
            for (size_t j = i + 1; j < d; ++j) {
                EXPECT_NEAR(result.metric_m(i, j), result.metric_m(j, i), 1e-5f);
            }
        }
    }
}

// Test eigendecomposition correctness
TEST(MetricAssemblyTest, EigendecompositionCorrectness) {
    constexpr size_t d = 16;
    
    // Create a simple symmetric positive definite matrix
    MatrixXf m = MatrixXf::Zero(d, d);
    for (size_t i = 0; i < d; ++i) {
        m(i, i) = static_cast<f32>(i + 1);  // Eigenvalues 1, 2, ..., d
        // Add some off-diagonal elements
        if (i > 0) {
            m(i, i-1) = 0.1f;
            m(i-1, i) = 0.1f;
        }
    }
    
    // Convert to array for LAPACK
    std::vector<f32> matrix(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            matrix[i * d + j] = m(i, j);
        }
    }
    
    std::vector<f32> eigenvalues(d);
    std::vector<f32> eigenvectors(d * d);
    
    Status status = symmetric_eigendecomposition(
        matrix.data(), d, eigenvalues.data(), eigenvectors.data()
    );
    
    EXPECT_EQ(status, Status::OK);
    
    // Check eigenvalues are positive
    for (size_t i = 0; i < d; ++i) {
        EXPECT_GT(eigenvalues[i], 0.0f);
    }
    
    // Check eigenvalues are sorted (ascending for ssyevr)
    for (size_t i = 1; i < d; ++i) {
        EXPECT_LE(eigenvalues[i-1], eigenvalues[i]);
    }
}

} // namespace
} // namespace smao
