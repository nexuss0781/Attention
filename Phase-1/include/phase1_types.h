#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace smao::phase1 {

// ============================================================================
// Type Aliases
// ============================================================================

using f32 = float;
using f64 = double;
using u32 = uint32_t;

// Matrix and vector types
using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Fixed-size types for compile-time dimension checking
template <int Rows, int Cols = Eigen::Dynamic>
using Matrix = Eigen::Matrix<f32, Rows, Cols>;

template <int Size = Eigen::Dynamic>
using Vector = Eigen::Vector<f32, Size>;

// ============================================================================
// Status Enum
// ============================================================================

enum class Status : int32_t {
    OK = 0,
    INPUT_NAN = -1,
    INPUT_INF = -2,
    METRIC_NOT_SPD = -3,
    METRIC_CONDITION_NUMBER_EXCEEDED = -4,
    EIGENDECOMPOSITION_FAILED = -5,
    INVALID_INPUT_SHAPE = -6,
    NUMERICAL_OVERFLOW = -7,
};

constexpr const char* status_to_string(Status s) {
    switch (s) {
        case Status::OK: return "OK";
        case Status::INPUT_NAN: return "INPUT_NAN";
        case Status::INPUT_INF: return "INPUT_INF";
        case Status::METRIC_NOT_SPD: return "METRIC_NOT_SPD";
        case Status::METRIC_CONDITION_NUMBER_EXCEEDED: return "METRIC_CONDITION_NUMBER_EXCEEDED";
        case Status::EIGENDECOMPOSITION_FAILED: return "EIGENDECOMPOSITION_FAILED";
        case Status::INVALID_INPUT_SHAPE: return "INVALID_INPUT_SHAPE";
        case Status::NUMERICAL_OVERFLOW: return "NUMERICAL_OVERFLOW";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Phase1Output Structure
// ============================================================================

struct Phase1Output {
    // Whitened coordinates (n x d)
    MatrixXf whitened_queries;
    MatrixXf whitened_keys;

    // Scalar weights (n-dimensional)
    VectorXf query_scales;  // a[i] = exp(||q_i||_2^2 / (2*sigma^2))
    VectorXf key_scales;    // w[j] = exp(||k_j||_2^2 / (2*sigma^2))

    // Bandwidth parameter
    f32 bandwidth;  // sigma^2 = sqrt(d_k)

    // Metric tensor M = LL^T + epsilon*I (d x d)
    MatrixXf metric;

    // Whitening operator W = M^{1/2} (d x d)
    MatrixXf whitening_operator;

    // Condition number kappa(M)
    f32 condition_number;

    // Status
    Status status;
};

// ============================================================================
// Numerical Constants & Tolerances
// ============================================================================

namespace constants {
    constexpr f32 EPSILON = 1e-6f;                    // SPD guarantee epsilon
    constexpr f32 MIN_EIGENVALUE = 1e-6f;             // Minimum eigenvalue threshold
    constexpr f32 MAX_CONDITION_NUMBER = 1e4f;        // Maximum kappa(M)
    constexpr f32 EXPONENT_MAX_SAFE = 80.0f;          // log(exp(88)) clipping
    constexpr f32 EXPONENT_MIN_SAFE = -80.0f;         // exp(-88) ~= 1.2e-38
    constexpr f32 DECOMPOSITION_ERROR_TOLERANCE = 1e-5f;  // Relative error tolerance
    constexpr f32 ISOMETRY_ERROR_TOLERANCE = 1e-6f;   // Isometry relative error
    constexpr f32 GRADIENT_TOLERANCE = 1e-4f;         // Finite-diff gradient tolerance
}

}  // namespace smao::phase1
