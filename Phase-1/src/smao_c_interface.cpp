#include "smao_phase1.h"
#include "phase1_forward.h"
#include "phase1_types.h"
#include <cstring>
#include <new>

using namespace smao::phase1;

extern "C" {

const char* smao_status_string(smao_status_t status) {
    switch (status) {
        case SMAO_OK: return "OK";
        case SMAO_INPUT_NAN: return "INPUT_NAN";
        case SMAO_INPUT_INF: return "INPUT_INF";
        case SMAO_METRIC_NOT_SPD: return "METRIC_NOT_SPD";
        case SMAO_METRIC_CONDITION_NUMBER_EXCEEDED: return "METRIC_CONDITION_NUMBER_EXCEEDED";
        case SMAO_EIGENDECOMPOSITION_FAILED: return "EIGENDECOMPOSITION_FAILED";
        case SMAO_INVALID_INPUT_SHAPE: return "INVALID_INPUT_SHAPE";
        case SMAO_NUMERICAL_OVERFLOW: return "NUMERICAL_OVERFLOW";
        default: return "UNKNOWN_STATUS";
    }
}

static smao_status_t status_to_c(Status s) {
    switch (s) {
        case Status::OK: return SMAO_OK;
        case Status::INPUT_NAN: return SMAO_INPUT_NAN;
        case Status::INPUT_INF: return SMAO_INPUT_INF;
        case Status::METRIC_NOT_SPD: return SMAO_METRIC_NOT_SPD;
        case Status::METRIC_CONDITION_NUMBER_EXCEEDED: return SMAO_METRIC_CONDITION_NUMBER_EXCEEDED;
        case Status::EIGENDECOMPOSITION_FAILED: return SMAO_EIGENDECOMPOSITION_FAILED;
        case Status::INVALID_INPUT_SHAPE: return SMAO_INVALID_INPUT_SHAPE;
        case Status::NUMERICAL_OVERFLOW: return SMAO_NUMERICAL_OVERFLOW;
        default: return SMAO_OK;
    }
}

smao_phase1_output_t* smao_phase1_forward(
    const float* Q, int32_t n, int32_t d,
    const float* K, int32_t n_k, int32_t d_k_input,
    const float* V, int32_t n_v, int32_t d_v,
    const float* L, int32_t d_L,
    uint32_t d_k
) {
    smao_phase1_output_t* c_output = nullptr;

    try {
        // Validate inputs
        if (n != n_k || n != n_v || d != d_k_input || d != d_L) {
            c_output = new smao_phase1_output_t();
            std::memset(c_output, 0, sizeof(smao_phase1_output_t));
            c_output->status = SMAO_INVALID_INPUT_SHAPE;
            return c_output;
        }

        // Map C arrays to Eigen matrices
        Eigen::Map<const MatrixXf> Q_eigen(const_cast<float*>(Q), n, d);
        Eigen::Map<const MatrixXf> K_eigen(const_cast<float*>(K), n_k, d);
        Eigen::Map<const MatrixXf> V_eigen(const_cast<float*>(V), n_v, d_v);
        Eigen::Map<const MatrixXf> L_eigen(const_cast<float*>(L), d_L, d_L);

        // Call C++ Phase1Forward
        Phase1Output cpp_output = phase1_forward(Q_eigen, K_eigen, V_eigen, L_eigen, d_k);

        // Allocate C output structure
        c_output = new smao_phase1_output_t();
        std::memset(c_output, 0, sizeof(smao_phase1_output_t));

        c_output->n = n;
        c_output->d = d;
        c_output->d_v = d_v;
        c_output->status = status_to_c(cpp_output.status);

        if (cpp_output.status != Status::OK) {
            return c_output;
        }

        // Allocate and copy whitened queries
        c_output->whitened_queries = new float[n * d];
        std::memcpy(c_output->whitened_queries, cpp_output.whitened_queries.data(), n * d * sizeof(float));

        // Allocate and copy whitened keys
        c_output->whitened_keys = new float[n * d];
        std::memcpy(c_output->whitened_keys, cpp_output.whitened_keys.data(), n * d * sizeof(float));

        // Allocate and copy scaling factors
        c_output->query_scales = new float[n];
        std::memcpy(c_output->query_scales, cpp_output.query_scales.data(), n * sizeof(float));

        c_output->key_scales = new float[n];
        std::memcpy(c_output->key_scales, cpp_output.key_scales.data(), n * sizeof(float));

        // Bandwidth
        c_output->bandwidth = cpp_output.bandwidth;

        // Allocate and copy metric
        c_output->metric = new float[d * d];
        std::memcpy(c_output->metric, cpp_output.metric.data(), d * d * sizeof(float));

        // Allocate and copy whitening operator
        c_output->whitening_operator = new float[d * d];
        std::memcpy(c_output->whitening_operator, cpp_output.whitening_operator.data(), d * d * sizeof(float));

        // Condition number
        c_output->condition_number = cpp_output.condition_number;

    } catch (const std::exception& e) {
        if (c_output) {
            smao_phase1_output_free(c_output);
        }
        c_output = new smao_phase1_output_t();
        std::memset(c_output, 0, sizeof(smao_phase1_output_t));
        c_output->status = SMAO_NUMERICAL_OVERFLOW;
    }

    return c_output;
}

void smao_phase1_output_free(smao_phase1_output_t* output) {
    if (!output) return;

    delete[] output->whitened_queries;
    delete[] output->whitened_keys;
    delete[] output->query_scales;
    delete[] output->key_scales;
    delete[] output->metric;
    delete[] output->whitening_operator;

    delete output;
}

smao_status_t smao_phase1_validate(
    const smao_phase1_output_t* output,
    const float* Q_original, int32_t n, int32_t d,
    const float* K_original
) {
    if (!output || !Q_original || !K_original) {
        return SMAO_INVALID_INPUT_SHAPE;
    }

    try {
        // Reconstruct C++ output
        Phase1Output cpp_output;
        cpp_output.status = Status::OK;
        cpp_output.bandwidth = output->bandwidth;
        cpp_output.condition_number = output->condition_number;

        // Map to Eigen
        cpp_output.whitened_queries = Eigen::Map<const MatrixXf>(
            const_cast<float*>(output->whitened_queries), n, d).eval();
        cpp_output.whitened_keys = Eigen::Map<const MatrixXf>(
            const_cast<float*>(output->whitened_keys), n, d).eval();
        cpp_output.query_scales = Eigen::Map<const VectorXf>(
            const_cast<float*>(output->query_scales), n).eval();
        cpp_output.key_scales = Eigen::Map<const VectorXf>(
            const_cast<float*>(output->key_scales), n).eval();
        cpp_output.metric = Eigen::Map<const MatrixXf>(
            const_cast<float*>(output->metric), d, d).eval();
        cpp_output.whitening_operator = Eigen::Map<const MatrixXf>(
            const_cast<float*>(output->whitening_operator), d, d).eval();

        // Map original Q, K
        Eigen::Map<const MatrixXf> Q_eigen(const_cast<float*>(Q_original), n, d);
        Eigen::Map<const MatrixXf> K_eigen(const_cast<float*>(K_original), n, d);

        // Call C++ validation
        Status validation_status = validate_phase1_output(cpp_output, Q_eigen, K_eigen);

        return status_to_c(validation_status);

    } catch (...) {
        return SMAO_NUMERICAL_OVERFLOW;
    }
}

}  // extern "C"
