/**
 * @file whiten_coordinates.h
 * @brief Coordinate whitening transformation
 *
 * Implements Algorithm 3: WhitenCoordinates
 * Applies whitening operator W to transform coordinates: X_tilde = X * W
 */

#ifndef SMAO_WHITEN_COORDINATES_H
#define SMAO_WHITEN_COORDINATES_H

#include "types.h"

namespace smao {

/**
 * @brief Whiten coordinates using whitening operator
 *
 * Implements Algorithm 3: WhitenCoordinates
 * Computes X_tilde = X * W where W is the whitening operator
 * (symmetric square root of metric M)
 *
 * Uses BLAS sgemm for efficient matrix multiplication.
 *
 * @param x Input coordinates (n x d, row-major)
 * @param whitening_w Whitening operator W (d x d, row-major, from metric_assembly)
 * @param n Number of tokens
 * @param d Dimension
 * @param x_whitened Output whitened coordinates (n x d, row-major, pre-allocated)
 * @return Status code
 */
Status whiten_coordinates(
    const f32* x,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* x_whitened
);

/**
 * @brief Whiten coordinates using Eigen (convenience wrapper)
 *
 * @param x Input coordinates (n x d)
 * @param whitening_w Whitening operator W (d x d)
 * @return MatrixXf Whitened coordinates (n x d)
 */
MatrixXf whiten_coordinates_eigen(
    const MatrixXf& x,
    const MatrixXf& whitening_w
);

/**
 * @brief Batch whiten coordinates for Q and K simultaneously
 *
 * Convenience function that whitens both query and key coordinates
 * in a single call.
 *
 * @param q Query matrix (n x d, row-major)
 * @param k Key matrix (n x d, row-major)
 * @param whitening_w Whitening operator W (d x d, row-major)
 * @param n Number of tokens
 * @param d Dimension
 * @param q_whitened Output whitened queries (n x d, row-major, pre-allocated)
 * @param k_whitened Output whitened keys (n x d, row-major, pre-allocated)
 * @return Status code
 */
Status whiten_coordinates_batch(
    const f32* q,
    const f32* k,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* q_whitened,
    f32* k_whitened
);

/**
 * @brief Verify whitening isometry
 *
 * Checks that ||W*x||^2 = x^T*M*x for random unit vectors.
 * Used for testing and validation.
 *
 * @param whitening_w Whitening operator W (d x d)
 * @param metric_m Metric M (d x d)
 * @param d Dimension
 * @param num_samples Number of random samples to test
 * @param max_relative_error Maximum allowed relative error
 * @return true if isometry holds within tolerance
 */
bool verify_whitening_isometry(
    const MatrixXf& whitening_w,
    const MatrixXf& metric_m,
    size_t d,
    size_t num_samples = 1000,
    f32 max_relative_error = 1e-6f
);

} // namespace smao

#endif // SMAO_WHITEN_COORDINATES_H
