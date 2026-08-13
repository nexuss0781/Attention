/*
 * @file c_api.cpp
 * @brief C API implementation for the Attention Phase 1 kernel.
 */

#include "smao_phase1/smao_phase1.h"
#include "smao_phase1/core/anisotropic_distance.h"
#include "smao_phase1/core/phase1_forward.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

using namespace smao;

namespace {

struct DistanceHandle {
    size_t d = 0;
    MatrixXf metric;
};

bool multiplication_fits(size_t a, size_t b) {
    return a == 0 || b <= std::numeric_limits<size_t>::max() / a;
}

smao_status_t status_to_c(Status status) {
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

bool precision_supported(smao_precision_t precision) {
    return precision == SMAO_F32;
}

Precision precision_from_c(smao_precision_t precision) {
    return precision == SMAO_F32 ? Precision::F32 : Precision::F16;
}

void release_buffers(smao_phase1_output_t* output) {
    if (output == nullptr) return;
    std::free(output->whitened_q);
    std::free(output->whitened_k);
    std::free(output->query_scales);
    std::free(output->key_weights);
    std::free(output->metric_m);
    std::free(output->whitening_w);
    output->whitened_q = nullptr;
    output->whitened_k = nullptr;
    output->query_scales = nullptr;
    output->key_weights = nullptr;
    output->metric_m = nullptr;
    output->whitening_w = nullptr;
}

} // namespace

extern "C" {

smao_status_t smao_phase1_forward(const smao_phase1_input_t* input,
                                   smao_phase1_output_t* output) {
    if (input == nullptr || output == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }

    // The public contract requires zero initialization before first use. We
    // only write a clean result after validation; callers must release an
    // existing successful result before reusing the structure.
    std::memset(output, 0, sizeof(*output));
    if (!precision_supported(input->precision)) {
        return SMAO_ERROR_INVALID_INPUT;
    }

    Phase1Input internal_input;
    internal_input.q = input->q;
    internal_input.k = input->k;
    internal_input.v = input->v;
    internal_input.l = input->l;
    internal_input.n = input->n;
    internal_input.d = input->d;
    internal_input.d_v = input->d_v;
    internal_input.precision = precision_from_c(input->precision);

    const Status validation = validate_phase1_input(internal_input);
    if (validation != Status::OK) {
        return status_to_c(validation);
    }

    Phase1Output internal_output = phase1_forward(internal_input);
    if (internal_output.status != Status::OK) {
        return status_to_c(internal_output.status);
    }

    if (!multiplication_fits(internal_output.n, internal_output.d) ||
        !multiplication_fits(internal_output.d, internal_output.d)) {
        return SMAO_ERROR_INVALID_INPUT;
    }

    const size_t matrix_count = internal_output.n * internal_output.d;
    const size_t vector_count = internal_output.n;
    const size_t metric_count = internal_output.d * internal_output.d;
    const size_t matrix_bytes = matrix_count * sizeof(float);
    const size_t vector_bytes = vector_count * sizeof(float);
    const size_t metric_bytes = metric_count * sizeof(float);

    output->whitened_q = static_cast<float*>(std::malloc(matrix_bytes));
    output->whitened_k = static_cast<float*>(std::malloc(matrix_bytes));
    output->query_scales = static_cast<float*>(std::malloc(vector_bytes));
    output->key_weights = static_cast<float*>(std::malloc(vector_bytes));
    output->metric_m = static_cast<float*>(std::malloc(metric_bytes));
    output->whitening_w = static_cast<float*>(std::malloc(metric_bytes));

    if (output->whitened_q == nullptr || output->whitened_k == nullptr ||
        output->query_scales == nullptr || output->key_weights == nullptr ||
        output->metric_m == nullptr || output->whitening_w == nullptr) {
        release_buffers(output);
        return SMAO_ERROR_ALLOCATION;
    }

    std::memcpy(output->whitened_q, internal_output.whitened_q.data(), matrix_bytes);
    std::memcpy(output->whitened_k, internal_output.whitened_k.data(), matrix_bytes);
    std::memcpy(output->query_scales, internal_output.query_scales.data(), vector_bytes);
    std::memcpy(output->key_weights, internal_output.key_weights.data(), vector_bytes);
    std::memcpy(output->metric_m, internal_output.metric_m.data(), metric_bytes);
    std::memcpy(output->whitening_w, internal_output.whitening_w.data(), metric_bytes);

    DistanceHandle* handle = static_cast<DistanceHandle*>(std::malloc(sizeof(DistanceHandle)));
    if (handle == nullptr) {
        release_buffers(output);
        return SMAO_ERROR_ALLOCATION;
    }
    try {
        new (handle) DistanceHandle();
        handle->d = internal_output.d;
        handle->metric = internal_output.metric_m;
    } catch (...) {
        std::free(handle);
        release_buffers(output);
        return SMAO_ERROR_ALLOCATION;
    }

    output->n = internal_output.n;
    output->d = internal_output.d;
    output->condition_number = internal_output.condition_number;
    output->sigma_squared = internal_output.sigma_squared;
    output->internal_handle = handle;
    return SMAO_OK;
}

smao_status_t smao_anisotropic_distance(void* raw_handle,
                                         const float* q,
                                         const float* k,
                                         float* distance_squared) {
    if (raw_handle == nullptr || q == nullptr || k == nullptr || distance_squared == nullptr) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    const auto* handle = static_cast<const DistanceHandle*>(raw_handle);
    if (handle->d == 0 || handle->metric.rows() != static_cast<Eigen::Index>(handle->d) ||
        handle->metric.cols() != static_cast<Eigen::Index>(handle->d)) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    for (size_t i = 0; i < handle->d; ++i) {
        if (!std::isfinite(q[i]) || !std::isfinite(k[i])) {
            return SMAO_ERROR_NAN_INPUT;
        }
    }
    const float value = anisotropic_distance_squared(q, k, handle->metric.data(), handle->d);
    if (!std::isfinite(value) || value < 0.0f) {
        return SMAO_ERROR_OVERFLOW;
    }
    *distance_squared = value;
    return SMAO_OK;
}

void smao_phase1_release(smao_phase1_output_t* output) {
    if (output == nullptr) return;
    release_buffers(output);
    if (output->internal_handle != nullptr) {
        auto* handle = static_cast<DistanceHandle*>(output->internal_handle);
        handle->~DistanceHandle();
        std::free(handle);
    }
    std::memset(output, 0, sizeof(*output));
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
    if (input == nullptr || !precision_supported(input->precision)) {
        return SMAO_ERROR_INVALID_INPUT;
    }
    Phase1Input internal_input;
    internal_input.q = input->q;
    internal_input.k = input->k;
    internal_input.v = input->v;
    internal_input.l = input->l;
    internal_input.n = input->n;
    internal_input.d = input->d;
    internal_input.d_v = input->d_v;
    internal_input.precision = precision_from_c(input->precision);
    return status_to_c(validate_phase1_input(internal_input));
}

} // extern "C"
