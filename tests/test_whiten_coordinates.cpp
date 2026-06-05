/**
 * @file test_whiten_coordinates.cpp
 * @brief Tests for coordinate whitening
 *
 * Test 1.2: Whitening Isometry verification
 */

#include <gtest/gtest.h>
#include "smao_phase1/core/whiten_coordinates.h"
#include "smao_phase1/core/metric_assembly.h"
#include <random>
#include <cmath>

namespace smao {
namespace {

// Test whitening isometry: ||W*x||^2 = x^T*M*x
TEST(WhitenCoordinatesTest, IsometryProperty) {
    constexpr size_t d = 64;
    constexpr size_t n_samples = 1000;
    
    // Generate random metric
    std::mt19937 gen(42);
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
    
    // Compute metric assembly
    MetricAssemblyResult result = metric_assembly(l.data(), d);
    
    EXPECT_EQ(result.status, Status::OK);
    
    f32 max_rel_error = 0.0f;
    
    // Test isometry on random unit vectors
    for (size_t sample = 0; sample < n_samples; ++sample) {
        // Generate random unit vector
        VectorXf x(d);
        for (size_t i = 0; i < d; ++i) {
            x(i) = dist(gen);
        }
        x.normalize();
        
        // Compute ||Wx||^2
        VectorXf wx = result.whitening_w * x;
        f32 wx_norm_sq = wx.squaredNorm();
        
        // Compute x^T M x
        f32 xMx = x.transpose() * result.metric_m * x;
        
        // Compute relative error
        f32 diff = std::abs(wx_norm_sq - xMx);
        f32 rel_error = (xMx > 1e-10f) ? diff / xMx : diff;
        
        max_rel_error = std::max(max_rel_error, rel_error);
    }
    
    // Isometry should hold to within 10^-6
    EXPECT_LT(max_rel_error, 1e-6f);
}

// Test whitening correctness with known input
TEST(WhitenCoordinatesTest, KnownInput) {
    // Simple 2D case
    constexpr size_t n = 3;
    constexpr size_t d = 2;
    
    // Input coordinates
    std::vector<f32> x = {
        1.0f, 0.0f,  // Point 1
        0.0f, 1.0f,  // Point 2
        1.0f, 1.0f   // Point 3
    };
    
    // Whitening matrix (scale by 2 in x, scale by 3 in y)
    std::vector<f32> w = {
        2.0f, 0.0f,
        0.0f, 3.0f
    };
    
    std::vector<f32> x_whitened(n * d);
    
    Status status = whiten_coordinates(
        x.data(), w.data(), n, d, x_whitened.data()
    );
    
    EXPECT_EQ(status, Status::OK);
    
    // Check expected results
    // Point 1: (1, 0) -> (2, 0)
    EXPECT_NEAR(x_whitened[0], 2.0f, 1e-5f);
    EXPECT_NEAR(x_whitened[1], 0.0f, 1e-5f);
    
    // Point 2: (0, 1) -> (0, 3)
    EXPECT_NEAR(x_whitened[2], 0.0f, 1e-5f);
    EXPECT_NEAR(x_whitened[3], 3.0f, 1e-5f);
    
    // Point 3: (1, 1) -> (2, 3)
    EXPECT_NEAR(x_whitened[4], 2.0f, 1e-5f);
    EXPECT_NEAR(x_whitened[5], 3.0f, 1e-5f);
}

// Test batch whitening
TEST(WhitenCoordinatesTest, BatchWhitening) {
    constexpr size_t n = 100;
    constexpr size_t d = 32;
    
    // Generate random Q and K
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    
    std::vector<f32> q(n * d);
    std::vector<f32> k(n * d);
    std::vector<f32> w(d * d, 0.0f);
    
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    
    // Create simple diagonal whitening matrix
    for (size_t i = 0; i < d; ++i) {
        w[i * d + i] = 1.0f + static_cast<f32>(i) * 0.1f;
    }
    
    std::vector<f32> q_whitened(n * d);
    std::vector<f32> k_whitened(n * d);
    
    Status status = whiten_coordinates_batch(
        q.data(), k.data(), w.data(), n, d,
        q_whitened.data(), k_whitened.data()
    );
    
    EXPECT_EQ(status, Status::OK);
    
    // Verify whitened coordinates are different from original
    bool different = false;
    for (size_t i = 0; i < n * d && !different; ++i) {
        if (std::abs(q_whitened[i] - q[i]) > 1e-6f) {
            different = true;
        }
    }
    EXPECT_TRUE(different);
}

} // namespace
} // namespace smao
