#pragma once

#include "phase1_types.h"

namespace smao::phase1 {

/**
 * Algorithm 2: MetricAssembly
 *
 * Assembles SPD metric tensor M from learned lower-triangular factor L:
 *   M = L*L^T + epsilon*I
 *
 * Performs eigendecomposition, enforces SPD constraints, computes condition number,
 * and derives symmetric matrix square root W = M^{1/2}.
 *
 * @param L Learned lower-triangular factor (d x d), with diag(L) > 0
 * @param out_M Output: metric tensor M (d x d)
 * @param out_W Output: whitening operator W = sqrt(M) (d x d)
 * @param out_kappa Output: condition number kappa(M)
 * @param out_eigenvalues Output: eigenvalues of M (optional diagnostic)
 * @return Status code
 */
Status metric_assembly(
    const MatrixXf& L,
    MatrixXf& out_M,
    MatrixXf& out_W,
    f32& out_kappa,
    VectorXf* out_eigenvalues = nullptr
);

/**
 * Compute symmetric matrix square root via eigendecomposition
 *
 * Given eigendecomposition M = U * Lambda * U^T, computes:
 *   sqrt(M) = U * sqrt(Lambda) * U^T
 *
 * @param U Eigenvectors (d x d)
 * @param eigenvalues Eigenvalues (d)
 * @return Symmetric matrix square root
 */
MatrixXf symmetric_matrix_sqrt(
    const MatrixXf& U,
    const VectorXf& eigenvalues
);

}  // namespace smao::phase1
