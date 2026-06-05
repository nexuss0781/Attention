#include <catch2/catch_test_macros.hpp>
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
TEST_CASE("Test1.5_NearSingularMetricRecovery", "[metric][stability]") {
    int d = 16;

    // Create L with very small diagonal entry (near-singular)
    MatrixXf L = MatrixXf::Identity(d, d);
    L(0, 0) = 1e-8f;

    MatrixXf M, W;
    float kappa;
    VectorXf eigenvalues;

    Status status = metric_assembly(L, M, W, kappa, &eigenvalues);
    REQUIRE(status == Status::OK);

    // Verify all eigenvalues >= MIN_EIGENVALUE
    for (int i = 0; i < d; ++i) {
        REQUIRE(eigenvalues(i) >= constants::MIN_EIGENVALUE - 1e-8f);
    }

    // Verify condition number constraint
    REQUIRE(kappa <= constants::MAX_CONDITION_NUMBER);

    // Verify M is still SPD
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(M);
    for (int i = 0; i < d; ++i) {
        REQUIRE(solver.eigenvalues()(i) > 0.0f);
    }
}

/**
 * Test with multiple near-zero diagonal entries
 */
TEST_CASE("Test_MetricAssembly_MultipleSmallEigenvalues", "[metric][stability]") {
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
    REQUIRE(status == Status::OK);

    // All eigenvalues should be safely bounded from below
    float lambda_min = eigenvalues.minCoeff();
    REQUIRE(lambda_min >= constants::MIN_EIGENVALUE - 1e-8f);

    // Condition number capped
    REQUIRE(kappa <= constants::MAX_CONDITION_NUMBER);
}
