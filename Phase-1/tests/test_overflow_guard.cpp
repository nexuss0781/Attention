#include <gtest/gtest.h>
#include "exact_decomposition.h"
#include "metric_assembly.h"
#include <Eigen/Dense>
#include <cmath>

using namespace smao::phase1;

/**
 * Test 1.4: Extreme Norm Overflow Guard
 *
 * Verify that extremely large norms are clipped safely and don't produce Inf/NaN.
 */
TEST(OverflowGuardTest, Test1_4_ExtremeNormOverflowGuard) {
    int n = 10;
    int d = 8;
    uint32_t d_k = d;

    MatrixXf Q = MatrixXf::Zero(n, d);
    MatrixXf K = MatrixXf::Zero(n, d);

    // Set row 0 with very large norm
    Q.row(0).setConstant(100.0f);  // ||Q[0]||^2 = 8 * 100^2 = 80000

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);

    if (status == Status::OK) {
        // Check that result is finite
        ASSERT_TRUE(std::isfinite(a(0)));
        ASSERT_TRUE(std::isfinite(w(0)));
        ASSERT_GT(a(0), 0.0f);

        // Value should be very large but not infinite
        ASSERT_LT(a(0), 1e38f);
    } else {
        // Overflow detected and handled
        ASSERT_EQ(status, Status::NUMERICAL_OVERFLOW);
    }
}

/**
 * Test safe exponent clipping
 */
TEST(OverflowGuardTest, SafeExpClipping) {
    // Test that exponent clipping prevents overflow
    float exp_arg_large = 100.0f;  // Would overflow without clipping
    float exp_arg_clipped = safe_exp_arg(exp_arg_large);

    ASSERT_LE(exp_arg_clipped, 80.0f);
    ASSERT_TRUE(std::isfinite(safe_expf(exp_arg_large)));

    // Test underflow protection
    float exp_arg_small = -100.0f;
    float exp_arg_clipped_small = safe_exp_arg(exp_arg_small);

    ASSERT_GE(exp_arg_clipped_small, -80.0f);
    ASSERT_TRUE(std::isfinite(safe_expf(exp_arg_small)));
}

/**
 * Test metric assembly with potential overflow scenarios
 */
TEST(OverflowGuardTest, MetricAssemblyStability) {
    int d = 16;

    // Create L with both very large and very small entries
    MatrixXf L = MatrixXf::Identity(d, d);
    L(0, 0) = 1e-4f;  // Small
    L(d - 1, d - 1) = 100.0f;  // Large

    MatrixXf M, W;
    float kappa;

    Status status = metric_assembly(L, M, W, kappa);
    ASSERT_EQ(status, Status::OK);

    // Verify all values are finite
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            ASSERT_TRUE(std::isfinite(M(i, j)));
            ASSERT_TRUE(std::isfinite(W(i, j)));
        }
    }

    // Condition number should be bounded despite wide range of L entries
    ASSERT_LE(kappa, 1e4f);
}
