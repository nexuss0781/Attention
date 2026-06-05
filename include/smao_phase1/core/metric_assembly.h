/**
 * @file metric_assembly.h
 * @brief Metric assembly and eigendecomposition
 *
 * Implements Algorithm 2: MetricAssembly
 * Computes M = L*L^T + epsilon*I, eigendecomposition, and whitening operator W.
 */

#ifndef SMAO_METRIC_ASSEMBLY_H
#define SMAO_METRIC_ASSEMBLY_H

#include "types.h"

namespace smao {

/**
 * @brief Metric assembly result structure
 */
struct MetricAssemblyResult {
    MatrixXf metric_m;          // d x d SPD matrix
    MatrixXf whitening_w;       // d x d symmetric square root
    f32 condition_number = 0.0f; // kappa(M)
    Status status = Status::OK;
};

/**
 * @brief Assemble metric from lower-triangular parameter L
 *
 * Implements Algorithm 2: MetricAssembly
 * 1. Computes M = L * L^T + epsilon * I
 * 2. Performs eigendecomposition via LAPACK ssyevr
 * 3. Enforces condition number bound via spectral projection
 * 4. Computes symmetric matrix square root W = U * Lambda^{1/2} * U^T
 *
 * @param l Lower triangular matrix L (d x d, row-major, lower triangular storage)
 * @param d Dimension
 * @param epsilon Regularization parameter (default: 1e-6)
 * @param condition_number_max Maximum condition number (default: 1e4)
 * @return MetricAssemblyResult containing M, W, and kappa(M)
 */
MetricAssemblyResult metric_assembly(
    const f32* l,
    size_t d,
    f32 epsilon = 1e-6f,
    f32 condition_number_max = 1e4f
);

/**
 * @brief Eigendecomposition of symmetric matrix
 *
 * Wrapper around LAPACK ssyevr for robust eigendecomposition.
 * Computes all eigenvalues and eigenvectors.
 *
 * @param matrix Input symmetric matrix (d x d, row-major, overwritten)
 * @param d Dimension
 * @param eigenvalues Output eigenvalues (size d, ascending order)
 * @param eigenvectors Output eigenvectors (d x d, row-major, column i is eigenvector i)
 * @return Status code
 */
Status symmetric_eigendecomposition(
    f32* matrix,  // Input/output: overwritten with eigenvectors
    size_t d,
    f32* eigenvalues,
    f32* eigenvectors
);

/**
 * @brief Compute symmetric matrix square root
 *
 * Computes W = U * Lambda^{1/2} * U^T where M = U * Lambda * U^T
 *
 * @param eigenvectors U matrix (d x d, row-major, column i is eigenvector i)
 * @param eigenvalues Lambda diagonal (size d, must be non-negative)
 * @param d Dimension
 * @param sqrt_matrix Output W matrix (d x d, row-major)
 * @return Status code
 */
Status symmetric_matrix_sqrt(
    const f32* eigenvectors,
    const f32* eigenvalues,
    size_t d,
    f32* sqrt_matrix
);

/**
 * @brief Enforce condition number bound via spectral projection
 *
 * Clips eigenvalues to ensure lambda_min >= lambda_max / kappa_max
 *
 * @param eigenvalues Input/output eigenvalues (size d)
 * @param d Dimension
 * @param kappa_max Maximum condition number
 * @return Actual condition number after projection
 */
f32 enforce_condition_number(
    f32* eigenvalues,
    size_t d,
    f32 kappa_max
);

/**
 * @brief Validate SPD properties of metric
 *
 * Checks symmetry, positive definiteness, and condition number.
 *
 * @param metric Input metric matrix (d x d, row-major)
 * @param d Dimension
 * @param epsilon Minimum eigenvalue threshold
 * @return Status code
 */
Status validate_spd(
    const f32* metric,
    size_t d,
    f32 epsilon = 1e-6f
);

} // namespace smao

#endif // SMAO_METRIC_ASSEMBLY_H
