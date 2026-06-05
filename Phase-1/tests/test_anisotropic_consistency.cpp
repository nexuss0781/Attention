#include <catch2/catch_test_macros.hpp>
#include "metric_assembly.h"
#include "whiten_coordinates.h"
#include "anisotropic_distance.h"
#include "numerical_stability.h"
#include <Eigen/Dense>

using namespace smao::phase1;

/**
 * Test 1.3: Anisotropic Kernel Consistency
 *
 * Verify that anisotropic distance in original space equals
 * Euclidean distance in whitened space.
 */
TEST_CASE("Test1.3_AnisotropicKernelConsistency", "[distance]") {
    int d = 64;

    // Generate SPD metric
    MatrixXf L = MatrixXf::Random(d, d).triangularView<Eigen::Lower>();
    for (int i = 0; i < d; ++i) {
        L(i, i) = std::abs(L(i, i)) + 1.0f;
    }

    MatrixXf M, W;
    float kappa;

    Status status = metric_assembly(L, M, W, kappa);
    REQUIRE(status == Status::OK);

    // Test on multiple random point pairs
    for (int test_iter = 0; test_iter < 100; ++test_iter) {
        VectorXf q = VectorXf::Random(d);
        VectorXf k = VectorXf::Random(d);

        // Anisotropic distance in original space
        float d_aniso_sq = anisotropic_squared_distance(q, k, M);

        // Whitened coordinates
        VectorXf q_w = W * q;
        VectorXf k_w = W * k;

        // Euclidean distance in whitened space
        float d_iso_sq = anisotropic_squared_distance_whitened(q_w, k_w);

        // Should be approximately equal
        float rel_error = std::abs(d_aniso_sq - d_iso_sq) / (std::abs(d_iso_sq) + 1e-10f);
        REQUIRE(rel_error < 1e-6f);
    }
}

/**
 * Test anisotropic distance properties
 */
TEST_CASE("Test_AnisotropicDistance_Properties", "[distance]") {
    int d = 16;

    MatrixXf L = MatrixXf::Identity(d, d);
    MatrixXf M, W;
    float kappa;

    Status status = metric_assembly(L, M, W, kappa);
    REQUIRE(status == Status::OK);

    VectorXf q = VectorXf::Random(d);
    VectorXf k = VectorXf::Random(d);

    // d_M^2(q, k) should be non-negative
    float d_sq = anisotropic_squared_distance(q, k, M);
    REQUIRE(d_sq >= -1e-6f);  // Allow small numerical error

    // d_M^2(q, q) should be near zero
    float d_self = anisotropic_squared_distance(q, q, M);
    REQUIRE(d_self < 1e-6f);

    // Symmetry: d_M^2(q, k) = d_M^2(k, q)
    float d_reverse = anisotropic_squared_distance(k, q, M);
    REQUIRE(std::abs(d_sq - d_reverse) < 1e-6f);
}
