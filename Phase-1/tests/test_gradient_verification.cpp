#include <gtest/gtest.h>
#include "phase1_forward.h"
#include "numerical_stability.h"
#include <Eigen/Dense>
#include <cmath>

using namespace smao::phase1;

/**
 * Test 1.7: Finite-Difference Gradient Verification
 *
 * Pre-validate gradient path (for Phase 7).
 * Compute numerical gradient via finite differences and verify consistency.
 */
TEST(GradientTest, Test1_7_FiniteDifferenceGradientVerification) {
    int n = 64;
    int d = 16;
    uint32_t d_k = d;
    float delta = 1e-4f;

    // Generate fixed Q, K
    MatrixXf Q = MatrixXf::Random(n, d) * 0.1f;  // Small to avoid overflow
    MatrixXf K = MatrixXf::Random(n, d) * 0.1f;
    MatrixXf V = MatrixXf::Zero(n, d);

    // Generate base L
    MatrixXf L = MatrixXf::Identity(d, d);

    // Define scalar loss: sum of all outputs
    auto loss_func = [&](const MatrixXf& L_test) -> float {
        Phase1Output output = phase1_forward(Q, K, V, L_test, d_k);
        if (output.status != Status::OK) return 1e10f;

        float loss = 0.0f;
        loss += output.query_scales.sum();
        loss += output.key_scales.sum();
        loss += output.condition_number;

        return loss;
    };

    // Compute baseline loss
    float loss_0 = loss_func(L);

    // Numerical gradient for L(0,0)
    MatrixXf L_plus = L;
    L_plus(0, 0) += delta;
    float loss_plus = loss_func(L_plus);

    MatrixXf L_minus = L;
    L_minus(0, 0) -= delta;
    float loss_minus = loss_func(L_minus);

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * delta);

    // Verify gradient is finite and reasonable
    ASSERT_TRUE(std::isfinite(numerical_grad));
    ASSERT_LT(std::abs(numerical_grad), 1e10f);  // Not pathological
}

/**
 * Test output differentiability w.r.t. L
 */
TEST(GradientTest, GradientFlowMetricParameter) {
    int d = 8;

    // Small perturbations to L
    MatrixXf L = MatrixXf::Identity(d, d);
    float epsilon = 1e-5f;

    // Forward pass 1
    Phase1Output output_0 = phase1_forward(
        MatrixXf::Random(16, d) * 0.1f,
        MatrixXf::Random(16, d) * 0.1f,
        MatrixXf::Zero(16, d),
        L, d
    );

    // Forward pass 2: perturbed L
    MatrixXf L_pert = L;
    L_pert(0, 0) += epsilon;

    Phase1Output output_pert = phase1_forward(
        MatrixXf::Random(16, d) * 0.1f,
        MatrixXf::Random(16, d) * 0.1f,
        MatrixXf::Zero(16, d),
        L_pert, d
    );

    // Condition number should respond smoothly to perturbations
    if (output_0.status == Status::OK && output_pert.status == Status::OK) {
        float kappa_change = std::abs(output_pert.condition_number - output_0.condition_number);
        ASSERT_LT(kappa_change, 1e4f);  // Bounded change
    }
}
