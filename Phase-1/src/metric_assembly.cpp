#include "metric_assembly.h"
#include "numerical_stability.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>

namespace smao::phase1 {

MatrixXf symmetric_matrix_sqrt(
    const MatrixXf& U,
    const VectorXf& eigenvalues
) {
    int d = U.rows();
    
    // Compute sqrt of eigenvalues
    VectorXf sqrt_eigenvalues = eigenvalues.array().sqrt();
    
    // Reconstruct: sqrt(M) = U * diag(sqrt(lambda)) * U^T
    // This is symmetric by construction
    MatrixXf Lambda_sqrt = sqrt_eigenvalues.asDiagonal();
    MatrixXf sqrt_M = U * Lambda_sqrt * U.transpose();
    
    return sqrt_M;
}

Status metric_assembly(
    const MatrixXf& L,
    MatrixXf& out_M,
    MatrixXf& out_W,
    f32& out_kappa,
    VectorXf* out_eigenvalues
) {
    // Input validation
    if (L.rows() != L.cols()) {
        return Status::INVALID_INPUT_SHAPE;
    }

    Status status = validate_matrix(L);
    if (status != Status::OK) return status;

    int d = L.rows();

    // 1. Compute M = L * L^T
    MatrixXf LLT = L * L.transpose();

    // 2. Add epsilon*I
    for (int i = 0; i < d; ++i) {
        LLT(i, i) += constants::EPSILON;
    }

    // 3. Eigendecomposition via Eigen's SelfAdjointEigenSolver
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(LLT);
    
    if (solver.info() != Eigen::Success) {
        return Status::EIGENDECOMPOSITION_FAILED;
    }

    MatrixXf U = solver.eigenvectors();
    VectorXf eigenvalues = solver.eigenvalues();

    // 4. Validate eigenvalues are positive
    Status spd_status = validate_spd_eigenvalues(eigenvalues);
    if (spd_status != Status::OK) return spd_status;

    // 5. Spectral projection: ensure all eigenvalues >= MIN_EIGENVALUE
    for (int i = 0; i < d; ++i) {
        if (eigenvalues(i) < constants::MIN_EIGENVALUE) {
            eigenvalues(i) = constants::MIN_EIGENVALUE;
        }
    }

    // 6. Compute condition number
    f32 lambda_max = eigenvalues.maxCoeff();
    f32 lambda_min = eigenvalues.minCoeff();
    out_kappa = lambda_max / lambda_min;

    // 7. Enforce condition number constraint
    if (out_kappa > constants::MAX_CONDITION_NUMBER) {
        enforce_condition_number(eigenvalues, constants::MAX_CONDITION_NUMBER);
        out_kappa = constants::MAX_CONDITION_NUMBER;
    }

    // 8. Recompute M from clipped eigenvalues to ensure consistency
    out_M = U * eigenvalues.asDiagonal() * U.transpose();

    // 9. Compute symmetric square root W
    out_W = symmetric_matrix_sqrt(U, eigenvalues);

    // Optional: return eigenvalues for diagnostics
    if (out_eigenvalues) {
        *out_eigenvalues = eigenvalues;
    }

    return Status::OK;
}

}  // namespace smao::phase1
