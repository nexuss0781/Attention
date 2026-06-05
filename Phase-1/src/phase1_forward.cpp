#include "phase1_forward.h"
#include "exact_decomposition.h"
#include "metric_assembly.h"
#include "whiten_coordinates.h"
#include "anisotropic_distance.h"
#include "numerical_stability.h"
#include <cmath>
#include <algorithm>

namespace smao::phase1 {

/**
 * Compute standard softmax attention scores for validation
 */
static MatrixXf compute_standard_attention_scores(
    const MatrixXf& Q,
    const MatrixXf& K,
    f32 sigma2
) {
    int n = Q.rows();
    MatrixXf scores(n, n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f32 dot_prod = Q.row(i).dot(K.row(j));
            f32 exp_arg = dot_prod / sigma2;
            exp_arg = std::clamp(exp_arg, constants::EXPONENT_MIN_SAFE, constants::EXPONENT_MAX_SAFE);
            scores(i, j) = std::exp(exp_arg);
        }
    }

    return scores;
}

/**
 * Compute decomposed attention scores: a[i] * w[j] * exp(-||q-k||_2^2 / (2*sigma^2))
 */
static MatrixXf compute_decomposed_attention_scores(
    const MatrixXf& Q_whitened,
    const MatrixXf& K_whitened,
    const VectorXf& a,
    const VectorXf& w,
    f32 sigma2
) {
    int n = Q_whitened.rows();
    MatrixXf scores(n, n);

    f32 denom = 2.0f * sigma2;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            VectorXf delta = Q_whitened.row(i) - K_whitened.row(j);
            f32 sq_dist = kahan_squared_norm(delta);

            f32 exp_arg = -sq_dist / denom;
            exp_arg = std::clamp(exp_arg, constants::EXPONENT_MIN_SAFE, constants::EXPONENT_MAX_SAFE);

            scores(i, j) = a(i) * w(j) * std::exp(exp_arg);
        }
    }

    return scores;
}

Phase1Output phase1_forward(
    const MatrixXf& Q,
    const MatrixXf& K,
    const MatrixXf& V,
    const MatrixXf& L,
    u32 d_k
) {
    Phase1Output output;
    output.status = Status::OK;

    // Input validation
    output.status = validate_matrix(Q);
    if (output.status != Status::OK) return output;
    output.status = validate_matrix(K);
    if (output.status != Status::OK) return output;
    output.status = validate_matrix(V);
    if (output.status != Status::OK) return output;
    output.status = validate_matrix(L);
    if (output.status != Status::OK) return output;

    // Algorithm 5: Phase1Forward
    // 1. MetricAssembly
    output.status = metric_assembly(L, output.metric, output.whitening_operator, output.condition_number);
    if (output.status != Status::OK) return output;

    // 2. WhitenCoordinates (Q)
    output.status = whiten_coordinates(Q, output.whitening_operator, output.whitened_queries);
    if (output.status != Status::OK) return output;

    // 3. WhitenCoordinates (K)
    output.status = whiten_coordinates(K, output.whitening_operator, output.whitened_keys);
    if (output.status != Status::OK) return output;

    // 4. ExactDecomposition (using whitened coordinates)
    output.status = exact_decomposition(
        output.whitened_queries,
        output.whitened_keys,
        d_k,
        output.query_scales,
        output.key_scales,
        output.bandwidth
    );
    if (output.status != Status::OK) return output;

    return output;
}

Status validate_phase1_output(
    const Phase1Output& output,
    const MatrixXf& Q_original,
    const MatrixXf& K_original
) {
    if (output.status != Status::OK) {
        return output.status;
    }

    int n = Q_original.rows();
    int d = Q_original.cols();

    // Test 1.1: Algebraic Equivalence
    // Compute standard attention and decomposed attention, compare
    MatrixXf standard_scores = compute_standard_attention_scores(Q_original, K_original, output.bandwidth);
    MatrixXf decomposed_scores = compute_decomposed_attention_scores(
        output.whitened_queries,
        output.whitened_keys,
        output.query_scales,
        output.key_scales,
        output.bandwidth
    );

    f32 max_rel_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (standard_scores(i, j) > 1e-10f) {
                f32 rel_err = std::abs(standard_scores(i, j) - decomposed_scores(i, j))
                             / standard_scores(i, j);
                max_rel_error = std::max(max_rel_error, rel_err);
            }
        }
    }

    if (max_rel_error > constants::DECOMPOSITION_ERROR_TOLERANCE) {
        return Status::NUMERICAL_OVERFLOW;
    }

    // Test 1.2: Whitening Isometry
    // Verify ||W*x||_2^2 = x^T*M*x for random unit vectors
    for (int iter = 0; iter < 100; ++iter) {
        VectorXf x = VectorXf::Random(d);
        x.normalize();

        VectorXf Wx = output.whitening_operator * x;
        f32 norm_Wx_sq = Wx.squaredNorm();
        f32 x_Mx = x.transpose() * output.metric * x;

        f32 iso_error = std::abs(norm_Wx_sq - x_Mx) / std::max(1e-10f, std::abs(x_Mx));
        if (iso_error > constants::ISOMETRY_ERROR_TOLERANCE) {
            return Status::METRIC_NOT_SPD;
        }
    }

    // Test 1.3: Metric Condition Number
    if (output.condition_number > constants::MAX_CONDITION_NUMBER) {
        return Status::METRIC_CONDITION_NUMBER_EXCEEDED;
    }

    return Status::OK;
}

}  // namespace smao::phase1
