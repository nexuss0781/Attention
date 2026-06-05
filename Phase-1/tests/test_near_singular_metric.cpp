#include <gtest/gtest.h>
#include "metric_assembly.h"
#include "numerical_stability.h"
#include <Eigen/Dense>

using namespace smao::phase1;

/**
 * Test 1.5: Near-Singular Metric Recovery
 *
 * Verify that near-singular metrics are projected to SPD with
 * bounded condition number.
 */
TEST(NearSingularMetricTest, Test1_5_NearSingularMetricRecovery) {
    int d = 16;

    // Create L with very small diagonal entry (near-singular)
    MatrixXf L = MatrixXf::Identity(d, d);
    L(0, 0) = 1e-8f;

    MatrixXf M, W;
    float kappa;
    VectorXf eigenvalues;

    Status status = metric_assembly(L, M, W, kappa, &eigenvalues);
    ASSERT_EQ(status, Status::OK);

    // Verify all eigenvalues >= MIN_EIGENVALUE
    for (int i = 0; i < d; ++i) {
        ASSERT_GE(eigenvalues(i), constants::MIN_EIGENVALUE - 1e-8f);
    }

    // Verify condition number constraint
    ASSERT_LE(kappa, constants::MAX_CONDITION_NUMBER);

    // Verify M is still SPD
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(M);
    for (int i = 0; i < d; ++i) {
        ASSERT_GT(solver.eigenvalues()(i), 0.0f);
    }
}

/**
 * Test with multiple near-zero diagonal entries
 */
TEST(NearSingularMetricTest, MultipleSmallEigenvalues) {
    int d = 16;

    // Create L with multiple small entries
    MatrixXf L = MatrixXf::Zero(d, d);
    for (int i = 0; i < d; ++i) {
        L(i, i) = (i < 4) ? 1e-8f : 1.0f;  // First 4 are tiny, rest are normal
    }

    MatrixXf M, W;
    float kappa;
    VectorXf eigenvalues;

    Status status = metric_assembly(L, M, W, kappa, &eigenvalues);
    ASSERT_EQ(status, Status::OK);

    // All eigenvalues should be safely bounded from below
    float lambda_min = eigenvalues.minCoeff();
    ASSERT_GE(lambda_min, constants::MIN_EIGENVALUE - 1e-8f);

    // Condition number capped
    ASSERT_LE(kappa, constants::MAX_CONDITION_NUMBER);
}
