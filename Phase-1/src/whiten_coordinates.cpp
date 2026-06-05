#include "whiten_coordinates.h"
#include "numerical_stability.h"

namespace smao::phase1 {

Status whiten_coordinates(
    const MatrixXf& X,
    const MatrixXf& W,
    MatrixXf& out_X_whitened
) {
    // Input validation
    if (X.cols() != W.rows()) {
        return Status::INVALID_INPUT_SHAPE;
    }
    if (W.rows() != W.cols()) {
        return Status::INVALID_INPUT_SHAPE;
    }

    Status status = validate_matrix(X);
    if (status != Status::OK) return status;
    status = validate_matrix(W);
    if (status != Status::OK) return status;

    // Algorithm 3: WhitenCoordinates
    // X̃ = X * W (BLAS gemm)
    out_X_whitened = X * W;

    return Status::OK;
}

}  // namespace smao::phase1
