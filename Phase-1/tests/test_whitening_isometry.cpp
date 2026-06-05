#include <gtest/gtest.h>
#include "metric_assembly.h"
#include "numerical_stability.h"
#include <Eigen/Dense>

using namespace smao::phase1;

/**
 * Test 1.2: Whitening Isometry
 *
 * Verify Theorem 3.2: For W = sqrt(M), all x satisfy ||W*x||_2^2 = x^T*M*x
 */
TEST(WhiteningIsometryTest, Test1_2_WhiteningIsometry) {
    int d = 64;

    // Generate random SPD metric via L*L^T
    MatrixXf L = MatrixXf::Random(d, d).triangularView<Eigen::Lower>();
    for (int i = 0; i < d; ++i) {
        L(i, i) = std::abs(L(i, i)) + 1.0f;  // Ensure positive diagonal
    }

    MatrixXf M, W;
    float kappa;

    Status status = metric_assembly(L, M, W, kappa);
    ASSERT_EQ(status, Status::OK);

    // Test isometry on 1000 random unit vectors
    float max_iso_error = 0.0f;

    for (int iter = 0; iter < 1000; ++iter) {
        VectorXf x = VectorXf::Random(d);
        x.normalize();

        // Compute ||W*x||_2^2
        VectorXf Wx = W * x;
        float norm_Wx_sq = Wx.squaredNorm();

        // Compute x^T*M*x
        float x_Mx = (x.transpose() * M * x)(0, 0);

        // Relative error
        float iso_error = std::abs(norm_Wx_sq - x_Mx) / (std::abs(x_Mx) + 1e-10f);
        max_iso_error = std::max(max_iso_error, iso_error);
    }

    ASSERT_LT(max_iso_error, 1e-6f);
}

/**
 * Test metric assembly properties
 */
TEST(WhiteningIsometryTest, SPDProperties) {
    int d = 32;

    MatrixXf L = MatrixXf::Random(d, d).triangularView<Eigen::Lower>();
    for (int i = 0; i < d; ++i) {
        L(i, i) = std::abs(L(i, i)) + 1.0f;
    }

    MatrixXf M, W;
    float kappa;
    VectorXf eigenvalues;

    Status status = metric_assembly(L, M, W, kappa, &eigenvalues);
    ASSERT_EQ(status, Status::OK);

    // Verify M is symmetric
    float max_asym = (M - M.transpose()).cwiseAbs().maxCoeff();
    ASSERT_LT(max_asym, 1e-5f);

    // Verify eigenvalues are positive
    for (int i = 0; i < d; ++i) {
        ASSERT_GT(eigenvalues(i), 1e-6f);
    }

    // Verify condition number is bounded
    ASSERT_LE(kappa, 1e4f);

    // Verify W is symmetric (W = U * sqrt(Lambda) * U^T)
    float max_W_asym = (W - W.transpose()).cwiseAbs().maxCoeff();
    ASSERT_LT(max_W_asym, 1e-5f);
}

/**
 * Test 1.5: Near-Singular Metric Recovery
 */
TEST(WhiteningIsometryTest, Test1_5_NearSingularMetricRecovery) {
    int d = 16;

    // Create L with one very small diagonal entry
    MatrixXf L = MatrixXf::Identity(d, d);
    L(0, 0) = 1e-8f;  // Very small, near-singular

    MatrixXf M, W;
    float kappa;
    VectorXf eigenvalues;

    Status status = metric_assembly(L, M, W, kappa, &eigenvalues);
    ASSERT_EQ(status, Status::OK);

    // Verify M has min eigenvalue >= 1e-6
    float lambda_min = eigenvalues.minCoeff();
    ASSERT_GE(lambda_min, 1e-6f - 1e-8f);  // Small tolerance for rounding

    // Verify condition number is capped
    ASSERT_LE(kappa, 1e4f);

    // Verify M is still SPD
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(M);
    for (int i = 0; i < d; ++i) {
        ASSERT_GT(solver.eigenvalues()(i), 0.0f);
    }
}
