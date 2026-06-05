/**
 * @file metric_assembly.cpp
 * @brief Implementation of metric assembly and eigendecomposition
 *
 * Implements Algorithm 2: MetricAssembly
 * Computes M = L*L^T + epsilon*I, eigendecomposition, and whitening operator W.
 */

#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/numerical_guards.h"
#include <cstring>
#include <algorithm>
#include <cmath>

// External LAPACK declarations
extern "C" {
    // Symmetric eigendecomposition (using bisection + inverse iteration)
    void ssyevr_(const char* jobz, const char* range, const char* uplo,
                 int* n, float* a, int* lda,
                 float* vl, float* vu, int* il, int* iu,
                 float* abstol, int* m, float* w,
                 float* z, int* ldz, int* isuppz,
                 float* work, int* lwork, int* iwork, int* liwork,
                 int* info);
}

namespace smao {

Status symmetric_eigendecomposition(
    f32* matrix,
    size_t d,
    f32* eigenvalues,
    f32* eigenvectors
) {
    if (matrix == nullptr || eigenvalues == nullptr || eigenvectors == nullptr) {
        return Status::InvalidInput;
    }
    if (d == 0 || d > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return Status::InvalidInput;
    }

    const int n = static_cast<int>(d);
    const int lda = n;
    const int ldz = n;
    
    // Query optimal workspace
    int lwork = -1;
    int liwork = -1;
    float work_query;
    int iwork_query;
    int m = 0;
    int info = 0;
    
    // Copy matrix to eigenvectors (LAPACK overwrites input)
    std::memcpy(eigenvectors, matrix, d * d * sizeof(f32));
    
    // Range parameters (compute all eigenvalues)
    const char jobz = 'V';  // Compute eigenvalues and eigenvectors
    const char range = 'A';  // All eigenvalues
    const char uplo = 'U';  // Upper triangle (we use symmetric storage)
    const float vl = 0.0f, vu = 0.0f;
    const int il = 1, iu = n;
    const float abstol = 0.0f;  // Use default tolerance
    
    // Query workspace
    ssyevr_(&jobz, &range, &uplo, &n, eigenvectors, &lda,
            &vl, &vu, &il, &iu, &abstol, &m, eigenvalues,
            nullptr, &ldz, nullptr, &work_query, &lwork,
            &iwork_query, &liwork, &info);
    
    if (info != 0) {
        return Status::Eigendecomposition;
    }
    
    lwork = static_cast<int>(work_query);
    liwork = iwork_query;
    
    // Allocate workspace
    std::vector<f32> work(lwork);
    std::vector<int> iwork(liwork);
    std::vector<int> isuppz(2 * n);
    
    // Re-copy matrix (was overwritten during query)
    std::memcpy(eigenvectors, matrix, d * d * sizeof(f32));
    
    // Compute eigendecomposition
    ssyevr_(&jobz, &range, &uplo, &n, eigenvectors, &lda,
            &vl, &vu, &il, &iu, &abstol, &m, eigenvalues,
            eigenvectors, &ldz, isuppz.data(), work.data(), &lwork,
            iwork.data(), &liwork, &info);
    
    if (info != 0) {
        return Status::Eigendecomposition;
    }
    
    if (m != n) {
        return Status::Eigendecomposition;
    }
    
    return Status::OK;
}

f32 enforce_condition_number_bound(
    f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0 || kappa_max <= 1.0f) {
        return 0.0f;
    }
    
    // Find min and max eigenvalues
    f32 lambda_min = eigenvalues[0];
    f32 lambda_max = eigenvalues[0];
    
    for (size_t i = 1; i < d; ++i) {
        if (eigenvalues[i] < lambda_min) lambda_min = eigenvalues[i];
        if (eigenvalues[i] > lambda_max) lambda_max = eigenvalues[i];
    }
    
    // Enforce minimum eigenvalue threshold
    f32 epsilon = LAMBDA_MIN_THRESHOLD;
    if (lambda_min < epsilon) {
        f32 shift = epsilon - lambda_min;
        for (size_t i = 0; i < d; ++i) {
            eigenvalues[i] += shift;
        }
        lambda_min = epsilon;
        lambda_max += shift;
    }
    
    // Compute current condition number
    f32 kappa = lambda_max / lambda_min;
    
    // If condition number exceeds bound, project eigenvalues
    if (kappa > kappa_max) {
        // Compute target minimum eigenvalue
        f32 lambda_min_target = lambda_max / kappa_max;
        
        // Clip eigenvalues to [lambda_min_target, lambda_max]
        for (size_t i = 0; i < d; ++i) {
            if (eigenvalues[i] < lambda_min_target) {
                eigenvalues[i] = lambda_min_target;
            }
        }
        
        // Update lambda_min
        lambda_min = lambda_min_target;
        kappa = kappa_max;
    }
    
    return kappa;
}

bool validate_condition_number(
    const f32* eigenvalues,
    size_t d,
    f32 kappa_max
) {
    if (eigenvalues == nullptr || d == 0) {
        return false;
    }
    
    f32 lambda_min = eigenvalues[0];
    f32 lambda_max = eigenvalues[0];
    
    for (size_t i = 1; i < d; ++i) {
        if (eigenvalues[i] < lambda_min) lambda_min = eigenvalues[i];
        if (eigenvalues[i] > lambda_max) lambda_max = eigenvalues[i];
    }
    
    if (lambda_min <= 0.0f) {
        return false;
    }
    
    f32 kappa = lambda_max / lambda_min;
    return kappa <= kappa_max;
}

Status symmetric_matrix_sqrt(
    const f32* eigenvectors,
    const f32* eigenvalues,
    size_t d,
    f32* sqrt_matrix
) {
    if (eigenvectors == nullptr || eigenvalues == nullptr || sqrt_matrix == nullptr) {
        return Status::InvalidInput;
    }
    if (d == 0) {
        return Status::InvalidInput;
    }
    
    // Check eigenvalues are non-negative
    for (size_t i = 0; i < d; ++i) {
        if (eigenvalues[i] < 0.0f) {
            return Status::InvalidInput;
        }
    }
    
    // Compute W = U * Lambda^{1/2} * U^T
    // W_{ij} = sum_k U_{ik} * sqrt(lambda_k) * U_{jk}
    
    // Note: eigenvectors is stored with column i being eigenvector i
    // In row-major: eigenvectors[i*d + j] is actually U_{ji} (i-th eigenvector, j-th component)
    
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f32 sum = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                // U_{ik} is component i of eigenvector k
                // eigenvectors[k*d + i] is component i of eigenvector k
                f32 u_ik = eigenvectors[k * d + i];
                f32 u_jk = eigenvectors[k * d + j];
                f32 sqrt_lambda_k = std::sqrt(eigenvalues[k]);
                sum += u_ik * sqrt_lambda_k * u_jk;
            }
            sqrt_matrix[i * d + j] = sum;
        }
    }
    
    return Status::OK;
}

MetricAssemblyResult metric_assembly(
    const f32* l,
    size_t d,
    f32 epsilon,
    f32 condition_number_max
) {
    MetricAssemblyResult result;
    
    if (l == nullptr) {
        result.status = Status::InvalidInput;
        return result;
    }
    if (d == 0) {
        result.status = Status::InvalidInput;
        return result;
    }
    if (condition_number_max < 1.0f) {
        result.status = Status::InvalidInput;
        return result;
    }
    
    // Allocate matrices
    result.metric_m.resize(d, d);
    result.whitening_w.resize(d, d);
    
    // Step 1: Compute M = L * L^T + epsilon * I
    // L is lower triangular
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f32 sum = 0.0f;
            // Sum over k: L[i,k] * L[j,k]
            // L is lower triangular, so only k <= min(i,j) contributes
            size_t k_max = (i < j) ? i : j;
            for (size_t k = 0; k <= k_max; ++k) {
                f32 l_ik = l[i * d + k];
                f32 l_jk = l[j * d + k];
                sum += l_ik * l_jk;
            }
            result.metric_m(i, j) = sum;
        }
        // Add epsilon to diagonal
        result.metric_m(i, i) += epsilon;
    }
    
    // Step 2: Eigendecomposition via LAPACK
    // Convert to row-major array for LAPACK
    std::vector<f32> matrix_copy(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            matrix_copy[i * d + j] = result.metric_m(i, j);
        }
    }
    
    std::vector<f32> eigenvalues(d);
    std::vector<f32> eigenvectors(d * d);
    
    Status eig_status = symmetric_eigendecomposition(
        matrix_copy.data(), d, eigenvalues.data(), eigenvectors.data()
    );
    
    if (eig_status != Status::OK) {
        result.status = eig_status;
        return result;
    }
    
    // Step 3: Enforce condition number bound
    result.condition_number = enforce_condition_number_bound(
        eigenvalues.data(), d, condition_number_max
    );
    
    // Check if condition number is still too large after projection
    if (result.condition_number > condition_number_max * 1.01f) {
        result.status = Status::ConditionNumber;
        return result;
    }
    
    // Step 4: Compute symmetric matrix square root W = U * Lambda^{1/2} * U^T
    Status sqrt_status = symmetric_matrix_sqrt(
        eigenvectors.data(), eigenvalues.data(), d,
        result.whitening_w.data()
    );
    
    if (sqrt_status != Status::OK) {
        result.status = sqrt_status;
        return result;
    }
    
    result.status = Status::OK;
    return result;
}

Status validate_spd(
    const f32* metric,
    size_t d,
    f32 epsilon
) {
    if (metric == nullptr) {
        return Status::InvalidInput;
    }
    if (d == 0) {
        return Status::InvalidInput;
    }
    
    // Check symmetry
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            f32 diff = std::abs(metric[i * d + j] - metric[j * d + i]);
            if (diff > epsilon) {
                return Status::InvalidInput;
            }
        }
    }
    
    // Check positive definiteness via Cholesky
    // This is a simplified check; full eigendecomposition would be more robust
    std::vector<f32> cholesky(d * d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            f32 sum = metric[i * d + j];
            for (size_t k = 0; k < j; ++k) {
                sum -= cholesky[i * d + k] * cholesky[j * d + k];
            }
            if (i == j) {
                if (sum <= epsilon) {
                    return Status::MetricSingular;
                }
                cholesky[i * d + j] = std::sqrt(sum);
            } else {
                cholesky[i * d + j] = sum / cholesky[j * d + j];
            }
        }
    }
    
    return Status::OK;
}

} // namespace smao
