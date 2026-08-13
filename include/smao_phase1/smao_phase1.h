/*
 * @file smao_phase1.h
 * @brief Public C API for the Attention Phase 1 numerical kernel.
 */

#ifndef SMAO_PHASE1_H
#define SMAO_PHASE1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef enum {
    SMAO_F32 = 0,
    SMAO_F16 = 1,
    SMAO_BF16 = 2
} smao_precision_t;

typedef struct {
    float* whitened_q;
    float* whitened_k;
    float* query_scales;
    float* key_weights;
    float* metric_m;
    float* whitening_w;
    float condition_number;
    float sigma_squared;
    size_t n;
    size_t d;
    void* internal_handle;
} smao_phase1_output_t;

typedef struct {
    const float* q;
    const float* k;
    const float* v;
    const float* l;
    size_t n;
    size_t d;
    size_t d_v;
    smao_precision_t precision;
} smao_phase1_input_t;

/*
 * The output structure must be zero-initialized before the first call. Before
 * reusing it, call smao_phase1_release. On failure, no output allocation is
 * retained and the structure remains safe to release.
 */
smao_status_t smao_phase1_forward(const smao_phase1_input_t* input,
                                   smao_phase1_output_t* output);

/* The handle must be obtained from a successful forward call. */
smao_status_t smao_anisotropic_distance(void* handle,
                                         const float* q,
                                         const float* k,
                                         float* distance_squared);

void smao_phase1_release(smao_phase1_output_t* output);
const char* smao_status_string(smao_status_t status);
smao_status_t smao_validate_input(const smao_phase1_input_t* input);

#ifdef __cplusplus
}
#endif

#endif /* SMAO_PHASE1_H */
