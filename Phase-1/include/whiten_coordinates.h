#pragma once

#include "phase1_types.h"

namespace smao::phase1 {

/**
 * Algorithm 3: WhitenCoordinates
 *
 * Applies whitening transformation to coordinate matrix:
 *   X̃ = X * W
 *
 * Complexity: O(n*d^2) arithmetic operations
 * Memory: O(n*d + d^2)
 *
 * @param X Input coordinates (n x d)
 * @param W Whitening operator (d x d), typically M^{1/2}
 * @param out_X_whitened Output: whitened coordinates (n x d)
 * @return Status code
 */
Status whiten_coordinates(
    const MatrixXf& X,
    const MatrixXf& W,
    MatrixXf& out_X_whitened
);

}  // namespace smao::phase1
