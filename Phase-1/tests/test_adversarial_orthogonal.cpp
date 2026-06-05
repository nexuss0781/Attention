#include <catch2/catch_test_macros.hpp>
#include "exact_decomposition.h"
#include "metric_assembly.h"
#include "whiten_coordinates.h"
#include "numerical_stability.h"
#include <Eigen/Dense>
#include <cmath>

using namespace smao::phase1;

/**
 * Test 1.6: Adversarial Orthogonal Inputs
 *
 * Verify stability when Q has mutually orthogonal rows with large norm.
 */
TEST_CASE("Test1.6_AdversarialOrthogonalInputs", "[stability]") {
    int d = 64;
    int n = d;  // Square case: n orthogonal vectors

    // Create orthogonal Q via QR decomposition
    MatrixXf Q_full = MatrixXf::Random(n, d);
    Eigen::HouseholderQR<MatrixXf> qr(Q_full);
    MatrixXf Q = qr.householderQ() * MatrixXf::Identity(n, d);

    // Scale to large norm
    Q *= 10.0f;

    MatrixXf K = MatrixXf::Zero(n, d);  // Keys are zero
    uint32_t d_k = d;

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);
    REQUIRE(status == Status::OK);

    // Verify no NaN/Inf
    for (int i = 0; i < n; ++i) {
        REQUIRE(std::isfinite(a(i)));
        REQUIRE(std::isfinite(w(i)));
    }

    // Values should be extremely large (but not infinite due to clipping)
    REQUIRE(a(0) > 0.0f);
    REQUIRE(a(0) < 1e38f);
}

/**
 * Test with unit-norm columns
 */
TEST_CASE("Test_NumericalStability_UnitNormColumns", "[stability]") {
    int d = 32;
    int n = 64;

    // Create Q with unit-norm rows
    MatrixXf Q = MatrixXf::Random(n, d);
    for (int i = 0; i < n; ++i) {
        Q.row(i).normalize();
    }

    MatrixXf K = Q.topRows(n);  // K same as Q
    uint32_t d_k = d;

    VectorXf a, w;
    float sigma2;

    Status status = exact_decomposition(Q, K, d_k, a, w, sigma2);
    REQUIRE(status == Status::OK);

    // All values should be close to unity (since ||q_i|| = ||k_j|| = 1)
    // a[i] = exp(1 / (2*sqrt(d_k)))
    // For d_k = 32, sqrt(32) ~ 5.66, so 1/(2*5.66) ~ 0.088
    // exp(0.088) ~ 1.092
    for (int i = 0; i < n; ++i) {
        REQUIRE(a(i) > 1.0f);
        REQUIRE(a(i) < 1.2f);
    }
}

/**
 * Test metric assembly with orthogonal basis
 */
TEST_CASE("Test_MetricAssembly_OrthogonalBasis", "[metric]") {
    int d = 16;

    // Create L from orthogonal basis (should be well-conditioned)
    MatrixXf L_full = MatrixXf::Random(d, d);
    Eigen::HouseholderQR<MatrixXf> qr(L_full);
    MatrixXf Q = qr.householderQ();

    // Use Q as lower-triangular approximation (just for test)
    MatrixXf L = Q.triangularView<Eigen::Lower>();
    for (int i = 0; i < d; ++i) {
        L(i, i) = std::abs(L(i, i)) + 1.0f;
    }

    MatrixXf M, W;
    float kappa;

    Status status = metric_assembly(L, M, W, kappa);
    REQUIRE(status == Status::OK);

    // Condition number should be reasonable
    REQUIRE(kappa < 100.0f);  // Tighter than MAX_CONDITION_NUMBER
}
