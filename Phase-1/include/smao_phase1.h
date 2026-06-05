#pragma once

/**
 * SMAO Phase 1: Kernel Geometry & Metric Space
 * Spectral Multipole Attention Operator (SMAO)
 *
 * Public C Interface for Phase 1 Forward Computation
 *
 * This header defines the public API for Phase 1. All users should include
 * this file and link against libsmao_phase1.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// ============================================================================
// Status Codes
// ============================================================================

typedef enum {
    SMAO_OK = 0,
    SMAO_INPUT_NAN = -1,
    SMAO_INPUT_INF = -2,
    SMAO_METRIC_NOT_SPD = -3,
    SMAO_METRIC_CONDITION_NUMBER_EXCEEDED = -4,
    SMAO_EIGENDECOMPOSITION_FAILED = -5,
    SMAO_INVALID_INPUT_SHAPE = -6,
    SMAO_NUMERICAL_OVERFLOW = -7,
} smao_status_t;

const char* smao_status_string(smao_status_t status);

// ============================================================================
// Phase1Output C Structure
// ============================================================================

typedef struct {
    // Dimensions
    int32_t n;    // Number of tokens
    int32_t d;    // Embedding dimension
    int32_t d_v;  // Value dimension

    // Whitened coordinates (row-major, size n*d)
    float* whitened_queries;
    float* whitened_keys;

    // Scalar weights (size n)
    float* query_scales;
    float* key_scales;

    // Bandwidth
    float bandwidth;

    // Metric M (row-major, d*d)
    float* metric;

    // Whitening operator W (row-major, d*d)
    float* whitening_operator;

    // Condition number
    float condition_number;

    // Status
    smao_status_t status;
} smao_phase1_output_t;

// ============================================================================
// Forward Computation
// ============================================================================

/**
 * Compute Phase 1 forward pass
 *
 * Inputs (all row-major float arrays):
 *   Q: n x d query embeddings
 *   K: n x d key embeddings
 *   V: n x d_v value embeddings (stored but not processed in Phase 1)
 *   L: d x d lower-triangular learned metric factor
 *   d_k: head dimension (typically = d)
 *
 * Returns allocated smao_phase1_output_t with all computed values.
 * Caller is responsible for freeing with smao_phase1_output_free().
 */
smao_phase1_output_t* smao_phase1_forward(
    const float* Q, int32_t n, int32_t d,
    const float* K, int32_t n_k, int32_t d_k_input,
    const float* V, int32_t n_v, int32_t d_v,
    const float* L, int32_t d_L,
    uint32_t d_k
);

/**
 * Free Phase 1 output
 */
void smao_phase1_output_free(smao_phase1_output_t* output);

// ============================================================================
// Validation
// ============================================================================

/**
 * Validate Phase 1 output against gate exit criteria
 *
 * @param output Phase1Output to validate
 * @param Q_original Original queries for reference
 * @param n Number of queries
 * @param d Dimension
 * @param K_original Original keys for reference
 * @return SMAO_OK if all criteria satisfied, else error code
 */
smao_status_t smao_phase1_validate(
    const smao_phase1_output_t* output,
    const float* Q_original, int32_t n, int32_t d,
    const float* K_original
);

#ifdef __cplusplus
}
#endif
