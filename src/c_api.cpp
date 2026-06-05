/**
 * @file c_api.cpp
 * @brief C API implementation for Phase 1
 *
 * Bridges C++ implementation with C linkage exports.
 */

#include "smao_phase1/smao_phase1.h"
#include "smao_phase1/core/types.h"
#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/whiten_coordinates.h"
#include "smao_phase1/core/anisotropic_distance.h"
#include "smao_phase1/core/numerical_guards.h"
#include "smao_phase1/core/phase1_forward.h"

#include <cstring>
#include <vector>
#include <memory>

using namespace smao;

// Convert internal Status to C smao_status_t
static smao_status_t status_to_c(Status status) {
    switch (status) {
        case Status::OK: return SMAO_OK;
        case Status::InvalidInput: return SMAO_ERROR_INVALID_INPUT;
        case Status::NaNInput: return SMAO_ERROR_NAN_INPUT;
        case Status::MetricSingular: return SMAO_ERROR_METRIC_SINGULAR;
        case Status::ConditionNumber: return SMAO_ERROR_CONDITION_NUMBER;
        case Status::Overflow: return SMAO_ERROR_OVERFLOW;
        case Status::Allocation: return SMAO_ERROR_ALLOCATION;
        case Status::Eigendecomposition: return SMAO_ERROR_EIGENDECOMPOSITION;
        default: return SMAO_ERROR_INVALID_INPUT;
    }
}

// Convert C smao_precision_t to internal Precision
static Precision precision_from_c(smao_precision_t precision) {
    switch (precision) {
        case SMAO_F32: return Precision::F32;
        case SMAO_F16: return Precision::F16;
        case SMAO_BF16: return Precision::BF16;
        default: return Precision::F32;
    }
}

extern "C" {

smao_status_t smao_phase1_forward(const smao_phase1_input_t* input,
                                   smao_phase1_output_t* output) {
    if (input == nullptr || output == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    
    // Clear output
    std::memset(output, 0, sizeof(smao_phase1_output_t));
    
    // Convert C input to internal input
    Phase1Input internal_input;
    internal_input.q = input->q;
    internal_input.k = input->k;
    internal_input.v = input->v;
    internal_input.l = input->l;
    internal_input.n = input->n;
    internal_input.d = input->d;
    internal_input.d_v = input->d_v;
    internal_input.precision = precision_from_c(input->precision);
    
    // Call internal implementation
    Phase1Output internal_output = phase1_forward(internal_input);
    
    // Copy internal output to C output
    output->n = internal_output.n;
    output->d = internal_output.d;
    output->sigma_squared = internal_output.sigma_squared;
    output->condition_number = internal_output.condition_number;
    
    // Allocate and copy matrices
    if (internal_output.n > 0 && internal_output.d > 0) {
        size_t matrix_size = internal_output.n * internal_output.d * sizeof(f32);
        size_t vector_size = internal_output.n * sizeof(f32);
        size_t metric_size = internal_output.d * internal_output.d * sizeof(f32);
        
        output->whitened_q = (f32*)std::malloc(matrix_size);
        output->whitened_k = (f32*)std::malloc(matrix_size);
        output->query_scales = (f32*)std::malloc(vector_size);
        output->key_weights = (f32*)std::malloc(vector_size);
        output->metric_m = (f32*)std::malloc(metric_size);
        output->whitening_w = (f32*)std::malloc(metric_size);
        
        if (output->whitened_q) {
            std::memcpy(output->whitened_q, internal_output.whitened_q.data(), matrix_size);
        }
        if (output->whitened_k) {
            std::memcpy(output->whitened_k, internal_output.whitened_k.data(), matrix_size);
        }
        if (output->query_scales) {
            std::memcpy(output->query_scales, internal_output.query_scales.data(), vector_size);
        }
        if (output->key_weights) {
            std::memcpy(output->key_weights, internal_output.key_weights.data(), vector_size);
        }
        if (output->metric_m) {
            std::memcpy(output->metric_m, internal_output.metric_m.data(), metric_size);
        }
        if (output->whitening_w) {
            std::memcpy(output->whitening_w, internal_output.whitening_w.data(), metric_size);
        }
    }
    
    return status_to_c(internal_output.status);
}

smao_status_t smao_anisotropic_distance(void* handle,
                                        const float* q,
                                        const float* k,
                                        float* distance_squared) {
    if (q == nullptr || k == nullptr || distance_squared == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    
    // For now, the handle is not used - we compute distance directly
    // In a full implementation, the handle would contain the metric and dimension
    
    // This is a placeholder - in actual use, the metric should be passed
    // through the handle or stored elsewhere
    *distance_squared = 0.0f;
    
    return SMAO_OK;
}

void smao_phase1_release(smao_phase1_output_t* output) {
    if (output == nullptr) {
        return;
    }
    
    if (output->whitened_q) {
        std::free(output->whitened_q);
        output->whitened_q = nullptr;
    }
    if (output->whitened_k) {
        std::free(output->whitened_k);
        output->whitened_k = nullptr;
    }
    if (output->query_scales) {
        std::free(output->query_scales);
        output->query_scales = nullptr;
    }
    if (output->key_weights) {
        std::free(output->key_weights);
        output->key_weights = nullptr;
    }
    if (output->metric_m) {
        std::free(output->metric_m);
        output->metric_m = nullptr;
    }
    if (output->whitening_w) {
        std::free(output->whitening_w);
        output->whitening_w = nullptr;
    }
    
    output->n = 0;
    output->d = 0;
    output->condition_number = 0.0f;
    output->sigma_squared = 0.0f;
}

const char* smao_status_string(smao_status_t status) {
    switch (status) {
        case SMAO_OK: return "OK";
        case SMAO_ERROR_INVALID_INPUT: return "Invalid input";
        case SMAO_ERROR_NAN_INPUT: return "NaN in input";
        case SMAO_ERROR_METRIC_SINGULAR: return "Metric is singular";
        case SMAO_ERROR_CONDITION_NUMBER: return "Condition number exceeds limit";
        case SMAO_ERROR_OVERFLOW: return "Overflow detected";
        case SMAO_ERROR_ALLOCATION: return "Memory allocation failed";
        case SMAO_ERROR_EIGENDECOMPOSITION: return "Eigendecomposition failed";
        default: return "Unknown error";
    }
}

smao_status_t smao_validate_input(const smao_phase1_input_t* input) {
    if (input == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    
    // Check pointers
    if (input->q == nullptr || input->k == nullptr || input->l == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    
    // Check dimensions
    if (input->n == 0 || input->d == 0) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    
    // Validate no NaN/Inf
    Status nan_check = validate_no_nan_inf(input->q, input->n * input->d);
    if (nan_check != Status::OK) {
        return status_to_c(nan_check);
    }
    
    nan_check = validate_no_nan_inf(input->k, input->n * input->d);
    if (nan_check != Status::OK) {
        return status_to_c(nan_check);
    }
    
    nan_check = validate_no_nan_inf(input->l, input->d * input->d);
    if (nan_check != Status::OK) {
        return status_to_c(nan_check);
    }
    
    return SMAO_OK;
}

} // extern "C"
