/**
 * @file smao_phase1.h
 * @brief Public C API for Phase 1: Kernel Geometry & Metric Space
 *
 * This header defines the C-linkage interface for the Spectral Multipole
 * Attention Operator (SMAO) Phase 1 implementation.
 *
 * @copyright (c) 2025
 * @license MIT
 */

#ifndef SMAO_PHASE1_H
#define SMAO_PHASE1_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for Phase 1 operations
 */
typedef enum {
    SMAO_OK = 0,
    SMAO_ERROR_INVALID_INPUT = 1,
    SMAO_ERROR_NAN_INPUT = 2,
    SMAO_ERROR_METRIC_SINGULAR = 3,
    SMAO_ERROR_CONDITION_NUMBER = 4,
    SMAO_ERROR_OVERFLOW = 5,
    SMAO_ERROR_ALLOCATION = 6,
    SMAO_ERROR_EIGENDECOMPOSITION = 7
} smao_status_t;

/**
 * @brief Floating point precision mode
 */
typedef enum {
    SMAO_F32 = 0,
    SMAO_F16 = 1,
    SMAO_BF16 = 2
} smao_precision_t;

/**
 * @brief Phase 1 output structure
 *
 * Contains all outputs from Phase 1 forward pass.
 * Memory layout is contiguous for cache efficiency.
 */
typedef struct {
    // Whitened coordinates (n x d)
    float* whitened_q;      // n x d matrix, row-major
    float* whitened_k;      // n x d matrix, row-major

    // Scaling factors
    float* query_scales;    // a_i = exp(||q_i||^2 / 2*sigma^2), size n
    float* key_weights;     // w_j = exp(||k_j||^2 / 2*sigma^2), size n

    // Metric and whitening operator
    float* metric_m;        // d x d SPD matrix, row-major
    float* whitening_w;     // d x d symmetric square root, row-major
    float condition_number; // kappa(M) = lambda_max / lambda_min

    // Bandwidth parameter
    float sigma_squared;  // sigma^2 = sqrt(d_k)

    // Dimensions
    size_t n;  // number of tokens
    size_t d;  // head dimension

    // Internal handle for distance primitive
    void* internal_handle;
} smao_phase1_output_t;

/**
 * @brief Phase 1 input configuration
 */
typedef struct {
    // Input tensors
    const float* q;  // n x d query matrix, row-major
    const float* k;  // n x d key matrix, row-major
    const float* v;  // n x d_v value matrix (for shape validation only in Phase 1)

    // Learned metric parameter L (lower triangular, d x d)
    const float* l;  // row-major, lower triangular

    // Dimensions
    size_t n;      // number of tokens
    size_t d;      // head dimension
    size_t d_v;    // value dimension

    // Precision mode
    smao_precision_t precision;
} smao_phase1_input_t;

/**
 * @brief Execute Phase 1 forward pass
 *
 * Implements the complete Phase 1 pipeline:
 * 1. Metric assembly from L parameter
 * 2. Coordinate whitening
 * 3. Exact Gaussian decomposition
 * 4. Numerical stability enforcement
 *
 * @param input Input configuration structure
 * @param output Output structure (must be pre-allocated)
 * @return Status code
 */
smao_status_t smao_phase1_forward(const smao_phase1_input_t* input,
                                   smao_phase1_output_t* output);

/**
 * @brief Compute anisotropic squared distance
 *
 * Computes d^2_M(q, k) = (q-k)^T M (q-k) using the metric from Phase 1.
 *
 * @param handle Internal handle from phase1_output
 * @param q Query vector (size d)
 * @param k Key vector (size d)
 * @param distance_squared Output distance squared
 * @return Status code
 */
smao_status_t smao_anisotropic_distance(void* handle,
                                         const float* q,
                                         const float* k,
                                         float* distance_squared);

/**
 * @brief Release resources associated with Phase 1 output
 *
 * @param output Phase 1 output structure
 */
void smao_phase1_release(smao_phase1_output_t* output);

/**
 * @brief Get error string for status code
 *
 * @param status Status code
 * @return Human-readable error string
 */
const char* smao_status_string(smao_status_t status);

/**
 * @brief Validate input parameters
 *
 * Checks for NaN, Inf, dimension consistency, and L constraints.
 *
 * @param input Input configuration
 * @return Status code
 */
smao_status_t smao_validate_input(const smao_phase1_input_t* input);

#ifdef __cplusplus
}
#endif

#endif // SMAO_PHASE1_H
