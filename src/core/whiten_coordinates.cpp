/**
 * @file whiten_coordinates.cpp
 * @brief Implementation of coordinate whitening
 *
 * Implements Algorithm 3: WhitenCoordinates
 * Applies whitening operator W to transform coordinates.
 */

#include "smao_phase1/core/whiten_coordinates.h"
#include <cstring>

// External BLAS declaration
extern "C" {
    // Single-precision matrix multiply: C = alpha*A*B + beta*C
    void sgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const float* alpha, const float* a, const int* lda,
                const float* b, const int* ldb,
                const float* beta, float* c, const int* ldc);
}

namespace smao {

Status whiten_coordinates(
    const f32* x,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* x_whitened
) {
    if (x == nullptr || whitening_w == nullptr || x_whitened == nullptr) {
        return Status::InvalidInput;
    }
    if (n == 0 || d == 0) {
        return Status::InvalidInput;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        d > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidInput;
    }
    
    int m = static_cast<int>(n);  // rows of A and C
    int k = static_cast<int>(d);  // cols of A and rows of B
    int n_cols = static_cast<int>(d);  // cols of B and C
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // X_whitened = X * W
    // X is m x k, W is k x n_cols, result is m x n_cols
    // In BLAS: C = alpha*A*B + beta*C
    // A = X (m x k), B = W (k x n_cols), C = X_whitened (m x n_cols)
    
    char transa = 'N';  // No transpose for A
    char transb = 'N';  // No transpose for B
    
    sgemm_(&transa, &transb, &m, &n_cols, &k,
           &alpha, x, &m,
           whitening_w, &k,
           &beta, x_whitened, &m);
    
    return Status::OK;
}

MatrixXf whiten_coordinates_eigen(
    const MatrixXf& x,
    const MatrixXf& whitening_w
) {
    // X_tilde = X * W
    return x * whitening_w;
}

Status whiten_coordinates_batch(
    const f32* q,
    const f32* k,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* q_whitened,
    f32* k_whitened
) {
    // Whiten queries
    Status status = whiten_coordinates(q, whitening_w, n, d, q_whitened);
    if (status != Status::OK) {
        return status;
    }
    
    // Whiten keys
    status = whiten_coordinates(k, whitening_w, n, d, k_whitened);
    if (status != Status::OK) {
        return status;
    }
    
    return Status::OK;
}

bool verify_whitening_isometry(
    const MatrixXf& whitening_w,
    const MatrixXf& metric_m,
    size_t d,
    size_t num_samples,
    f32 max_relative_error
) {
    if (d == 0 || num_samples == 0) {
        return false;
    }
    
    // Generate random unit vectors and verify isometry
    for (size_t sample = 0; sample < num_samples; ++sample) {
        // Generate random vector
        VectorXf x(d);
        for (size_t i = 0; i < d; ++i) {
            // Simple pseudo-random based on sample and index
            x(i) = static_cast<f32>(std::sin(sample * 1000.0 + i * 0.5) * 0.5 + 0.5);
        }
        
        // Normalize to unit vector
        f32 norm = x.norm();
        if (norm > 1e-10f) {
            x /= norm;
        } else {
            continue;  // Skip degenerate case
        }
        
        // Compute ||Wx||^2
        VectorXf wx = whitening_w * x;
        f32 wx_norm_sq = wx.squaredNorm();
        
        // Compute x^T M x
        f32 xMx = x.transpose() * metric_m * x;
        
        // Check relative error
        f32 diff = std::abs(wx_norm_sq - xMx);
        f32 relative_error = (xMx > 1e-10f) ? diff / xMx : diff;
        
        if (relative_error > max_relative_error) {
            return false;
        }
    }
    
    return true;
}

} // namespace smao
