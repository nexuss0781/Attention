#include "exact_decomposition.h"
#include "numerical_stability.h"
#include <cmath>
#include <algorithm>

namespace smao::phase1 {

Status exact_decomposition(
    const MatrixXf& Q,
    const MatrixXf& K,
    u32 d_k,
    VectorXf& out_a,
    VectorXf& out_w,
    f32& out_sigma2
) {
    // Input validation
    if (Q.rows() != K.rows()) {
        return Status::INVALID_INPUT_SHAPE;
    }
    if (static_cast<u32>(Q.cols()) != d_k) {
        return Status::INVALID_INPUT_SHAPE;
    }

    Status status = validate_matrix(Q);
    if (status != Status::OK) return status;
    status = validate_matrix(K);
    if (status != Status::OK) return status;

    int n = Q.rows();
    int d = Q.cols();

    // Allocate output vectors
    out_a.resize(n);
    out_w.resize(n);

    // Algorithm 1: ExactDecomposition
    // 1. sigma^2 = sqrt(d_k)
    out_sigma2 = std::sqrt(static_cast<f32>(d_k));

    f32 denom = 2.0f * out_sigma2;  // 2 * sigma^2

    // 2. Compute query scaling factors a[i] = exp(||q_i||_2^2 / (2*sigma^2))
    for (int i = 0; i < n; ++i) {
        f32 sq_norm_q = kahan_squared_norm(Q.row(i));

        // Safe exponent range clipping
        f32 exp_arg_q = sq_norm_q / denom;
        if (exp_arg_q > constants::EXPONENT_MAX_SAFE) {
            return Status::NUMERICAL_OVERFLOW;
        }
        exp_arg_q = std::max(exp_arg_q, constants::EXPONENT_MIN_SAFE);

        out_a(i) = std::exp(exp_arg_q);
    }

    // 3. Compute key scaling factors w[j] = exp(||k_j||_2^2 / (2*sigma^2))
    for (int j = 0; j < n; ++j) {
        f32 sq_norm_k = kahan_squared_norm(K.row(j));

        // Safe exponent range clipping
        f32 exp_arg_k = sq_norm_k / denom;
        if (exp_arg_k > constants::EXPONENT_MAX_SAFE) {
            return Status::NUMERICAL_OVERFLOW;
        }
        exp_arg_k = std::max(exp_arg_k, constants::EXPONENT_MIN_SAFE);

        out_w(j) = std::exp(exp_arg_k);
    }

    return Status::OK;
}

}  // namespace smao::phase1
