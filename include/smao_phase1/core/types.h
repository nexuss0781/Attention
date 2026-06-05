/**
 * @file types.h
 * @brief Core type definitions for SMAO Phase 1
 */

#ifndef SMAO_CORE_TYPES_H
#define SMAO_CORE_TYPES_H

#include <cstddef>
#include <cstdint>
#include <array>
#include <vector>
#include <Eigen/Dense>

namespace smao {

// Floating point precision aliases
using f32 = float;
using f64 = double;

// Fixed-size vector types for small dimensions
template <size_t N>
using Vec = std::array<f32, N>;

// Dynamic vector type
using VecDyn = std::vector<f32>;

// Eigen matrix types with custom allocator for alignment
using MatrixXf = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXf = Eigen::Matrix<f32, Eigen::Dynamic, 1>;

using MatrixXd = Eigen::Matrix<f64, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::Matrix<f64, Eigen::Dynamic, 1>;

// Status codes for internal use
enum class Status : int32_t {
    OK = 0,
    InvalidInput = 1,
    NaNInput = 2,
    MetricSingular = 3,
    ConditionNumber = 4,
    Overflow = 5,
    Allocation = 6,
    Eigendecomposition = 7
};

// Precision modes
enum class Precision : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2
};

// Input configuration structure
struct Phase1Input {
    const f32* q = nullptr;      // n x d query matrix, row-major
    const f32* k = nullptr;      // n x d key matrix, row-major
    const f32* v = nullptr;      // n x d_v value matrix
    const f32* l = nullptr;      // d x d lower triangular metric parameter
    
    size_t n = 0;      // number of tokens
    size_t d = 0;      // head dimension
    size_t d_v = 0;    // value dimension
    
    Precision precision = Precision::F32;
    
    // Tolerance parameters
    f32 epsilon = 1e-6f;           // Regularization for SPD
    f32 condition_number_max = 1e4f; // Maximum condition number
    f32 log_clip_min = -80.0f;       // Minimum log-space value
    f32 log_clip_max = 80.0f;        // Maximum log-space value
};

// Output structure
struct Phase1Output {
    // Whitened coordinates
    MatrixXf whitened_q;  // n x d
    MatrixXf whitened_k;  // n x d
    
    // Scaling factors
    VectorXf query_scales;   // a_i, size n
    VectorXf key_weights;    // w_j, size n
    
    // Metric and whitening operator
    MatrixXf metric_m;       // d x d SPD matrix
    MatrixXf whitening_w;    // d x d symmetric square root
    f32 condition_number = 0.0f;  // kappa(M)
    
    // Bandwidth parameter
    f32 sigma_squared = 0.0f;  // sigma^2 = sqrt(d_k)
    
    // Dimensions
    size_t n = 0;
    size_t d = 0;
    
    // Status
    Status status = Status::OK;
};

// Distance primitive function pointer type
using DistancePrimitiveFn = f32(*)(const f32* q, const f32* k, const MatrixXf& metric, size_t d);

// Helper functions
inline const char* status_to_string(Status status) {
    switch (status) {
        case Status::OK: return "OK";
        case Status::InvalidInput: return "InvalidInput";
        case Status::NaNInput: return "NaNInput";
        case Status::MetricSingular: return "MetricSingular";
        case Status::ConditionNumber: return "ConditionNumber";
        case Status::Overflow: return "Overflow";
        case Status::Allocation: return "Allocation";
        case Status::Eigendecomposition: return "Eigendecomposition";
        default: return "Unknown";
    }
}

} // namespace smao

#endif // SMAO_CORE_TYPES_H
