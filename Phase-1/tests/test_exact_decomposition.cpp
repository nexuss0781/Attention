#include <gtest/gtest.h>
#include "exact_decomposition.h"
#include "numerical_stability.h"
#include <Eigen/Dense>
#include <cmath>

using namespace smao::phase1;

/**
 * Test 1.1: Algebraic Equivalence (The Core Invariant)
 *
 * Verify Theorem 1.1: For whitened coordinates Q, K and sigma^2 = sqrt(d_k),
 *   exp(<q, k> / sigma^2) = a[i] * w[j] * exp(-||q-k||_2^2 / (2*sigma^2))
 */
TEST(DecompositionTest, Test1_1_AlgebraicEquivalence) {
    int n = 512;
    int d = 64;
    uint32_t d_k = d;

    // Generate random Q, K from N(0, 1)
    MatrixXf Q = MatrixXf::Random(n, d);
    MatrixXf K = MatrixXf::Random(n, d);

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);
    ASSERT_EQ(status, Status::OK);
    ASSERT_EQ(a.size(), n);
    ASSERT_EQ(w.size(), n);

    // Compute standard attention: exp(<q_i, k_j> / sigma^2)
    // Compute decomposed attention: a[i] * w[j] * exp(-||q_i - k_j||_2^2 / (2*sigma^2))

    float max_rel_error = 0.0f;
    float denom_2 = 2.0f * sigma2;

    for (int i = 0; i < std::min(n, 10); ++i) {  // Check subset for speed
        for (int j = 0; j < std::min(n, 10); ++j) {
            // Standard attention
            float dot_prod = Q.row(i).dot(K.row(j));
            float exp_arg_standard = dot_prod / sigma2;
            exp_arg_standard = std::clamp(exp_arg_standard, -80.0f, 80.0f);
            float standard_score = std::exp(exp_arg_standard);

            // Decomposed attention
            VectorXf delta = Q.row(i) - K.row(j);
            float sq_dist = delta.squaredNorm();
            float exp_arg_decomp = -sq_dist / denom_2;
            exp_arg_decomp = std::clamp(exp_arg_decomp, -80.0f, 80.0f);
            float decomp_score = a(i) * w(j) * std::exp(exp_arg_decomp);

            float rel_error = std::abs(standard_score - decomp_score) / (std::abs(standard_score) + 1e-10f);
            max_rel_error = std::max(max_rel_error, rel_error);
        }
    }

    ASSERT_LT(max_rel_error, 1e-5f);
}

/**
 * Test input validation
 */
TEST(DecompositionTest, InputValidation) {
    int n = 10;
    int d = 8;
    uint32_t d_k = d;

    MatrixXf Q = MatrixXf::Random(n, d);
    MatrixXf K = MatrixXf::Random(n, d);

    VectorXf a, w;
    float sigma2;

    // Valid case
    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);
    ASSERT_EQ(status, Status::OK);

    // Dimension mismatch
    MatrixXf K_bad = MatrixXf::Random(n + 1, d);
    status = exact_decomposition(Q, K_bad, d_k, a, w, sigma2);
    ASSERT_EQ(status, Status::INVALID_INPUT_SHAPE);
}

/**
 * Test overflow guard
 */
TEST(DecompositionTest, Test1_4_OverflowGuard) {
    int n = 10;
    int d = 8;
    uint32_t d_k = d;

    MatrixXf Q = MatrixXf::Zero(n, d);
    MatrixXf K = MatrixXf::Zero(n, d);

    // Set one row with very large norm to trigger overflow condition
    Q.row(0).setConstant(100.0f);  // ||Q[0]|| = sqrt(8)*100 ~ 282

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);

    // Should either return overflow or safe clipped values
    if (status == Status::OK) {
        // Verify results are finite
        ASSERT_TRUE(std::isfinite(a(0)));
        ASSERT_GT(a(0), 0.0f);
    } else {
        ASSERT_EQ(status, Status::NUMERICAL_OVERFLOW);
    }
}
