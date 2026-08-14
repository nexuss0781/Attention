/*
 * @file phase1_forward.cpp
 * @brief Phase 1 forward pass orchestrator.
 */

#include "smao_phase1/core/phase1_forward.h"
#include "smao_phase1/core/anisotropic_distance.h"
#include "smao_phase1/core/exact_decomposition.h"
#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/numerical_guards.h"
#include "smao_phase1/core/whiten_coordinates.h"

#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Eigenvalues>

namespace smao {

namespace {

bool multiplication_fits(size_t a, size_t b) {
    return a == 0 || b <= std::numeric_limits<size_t>::max() / a;
}

Status validate_forward_fast(const Phase1Input& input) {
    if (input.q == nullptr || input.k == nullptr || input.v == nullptr || input.l == nullptr ||
        input.n == 0 || input.d == 0 || input.d_v == 0 || input.precision != Precision::F32 ||
        !std::isfinite(input.epsilon) || input.epsilon < 0.0f ||
        !std::isfinite(input.condition_number_max) || input.condition_number_max <= 1.0f ||
        !std::isfinite(input.log_clip_min) || !std::isfinite(input.log_clip_max) ||
        input.log_clip_min >= input.log_clip_max ||
        !multiplication_fits(input.n, input.d) ||
        !multiplication_fits(input.n, input.d_v) ||
        !multiplication_fits(input.d, input.d)) {
        return Status::InvalidInput;
    }
    for (size_t i = 0; i < input.d; ++i) {
        const f32 diagonal = input.l[i * input.d + i];
        if (!std::isfinite(diagonal) || !(diagonal > 0.0f)) return Status::InvalidInput;
        for (size_t j = i + 1; j < input.d; ++j) {
            if (!std::isfinite(input.l[i * input.d + j]) ||
                std::abs(input.l[i * input.d + j]) > 1e-6f) return Status::InvalidInput;
        }
    }
    return Status::OK;
}

f32 whitening_metric_residual(const Phase1Output& output) {
    if (output.metric_m.rows() != static_cast<Eigen::Index>(output.d) ||
        output.metric_m.cols() != static_cast<Eigen::Index>(output.d) ||
        output.whitening_w.rows() != static_cast<Eigen::Index>(output.d) ||
        output.whitening_w.cols() != static_cast<Eigen::Index>(output.d) ||
        !output.metric_m.allFinite() || !output.whitening_w.allFinite()) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    f32 maximum_relative_error = 0.0f;
    for (size_t i = 0; i < output.d; ++i) {
        for (size_t j = 0; j < output.d; ++j) {
            f64 reconstructed = 0.0;
            for (size_t k = 0; k < output.d; ++k) {
                reconstructed += static_cast<f64>(output.whitening_w(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(i))) *
                                 static_cast<f64>(output.whitening_w(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(j)));
            }
            const f64 expected = static_cast<f64>(output.metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
            const f64 error = std::abs(reconstructed - expected) / std::max(1.0, std::abs(expected));
            maximum_relative_error = std::max(maximum_relative_error, static_cast<f32>(error));
        }
    }
    return maximum_relative_error;
}

f32 distance_primitive_adapter(
    const f32* q,
    const f32* k,
    const MatrixXf& metric,
    size_t d
) {
    if (metric.rows() != static_cast<Eigen::Index>(d) ||
        metric.cols() != static_cast<Eigen::Index>(d)) {
        return std::numeric_limits<f32>::quiet_NaN();
    }
    return anisotropic_distance_squared(q, k, metric.data(), d);
}

} // namespace

Status validate_phase1_input(const Phase1Input& input) {
    if (input.q == nullptr || input.k == nullptr || input.v == nullptr || input.l == nullptr) {
        return Status::InvalidInput;
    }
    if (input.n == 0 || input.d == 0 || input.d_v == 0) {
        return Status::InvalidInput;
    }
    if (input.precision != Precision::F32) {
        return Status::InvalidInput;
    }
    if (!std::isfinite(input.epsilon) || input.epsilon < 0.0f ||
        !std::isfinite(input.condition_number_max) || input.condition_number_max <= 1.0f ||
        !std::isfinite(input.log_clip_min) || !std::isfinite(input.log_clip_max) ||
        input.log_clip_min >= input.log_clip_max) {
        return Status::InvalidInput;
    }
    if (!multiplication_fits(input.n, input.d) ||
        !multiplication_fits(input.n, input.d_v) ||
        !multiplication_fits(input.d, input.d)) {
        return Status::InvalidInput;
    }

    Status status = validate_no_nan_inf(input.q, input.n * input.d);
    if (status != Status::OK) return status;
    status = validate_no_nan_inf(input.k, input.n * input.d);
    if (status != Status::OK) return status;
    status = validate_no_nan_inf(input.v, input.n * input.d_v);
    if (status != Status::OK) return status;
    status = validate_no_nan_inf(input.l, input.d * input.d);
    if (status != Status::OK) return status;

    for (size_t i = 0; i < input.d; ++i) {
        const f32 diagonal = input.l[i * input.d + i];
        if (!(diagonal > 0.0f)) {
            return Status::InvalidInput;
        }
        for (size_t j = i + 1; j < input.d; ++j) {
            if (std::abs(input.l[i * input.d + j]) > 1e-6f) {
                return Status::InvalidInput;
            }
        }
    }
    return Status::OK;
}

Status phase1_forward_into(const Phase1Input& input, Phase1Output& output) {
    output.n = input.n;
    output.d = input.d;

    // Q and K finite-value validation is fused with the production whitening
    // kernel; this avoids scanning hundreds of megabytes twice before work.
    Status status = validate_forward_fast(input);
    if (status != Status::OK) {
        output.status = status;
        return output.status;
    }

    MetricAssemblyResult metric_result = metric_assembly(
        input.l, input.d, input.epsilon, input.condition_number_max);
    if (metric_result.status != Status::OK) {
        output.status = metric_result.status;
        return output.status;
    }

    output.metric_m = std::move(metric_result.metric_m);
    output.whitening_w = std::move(metric_result.whitening_w);
    output.condition_number = metric_result.condition_number;

    output.whitened_q.resize(static_cast<Eigen::Index>(input.n), static_cast<Eigen::Index>(input.d));
    output.whitened_k.resize(static_cast<Eigen::Index>(input.n), static_cast<Eigen::Index>(input.d));
    status = whiten_coordinates_pair_prevalidated(
        input.q, input.k, output.whitening_w.data(), input.n, input.d,
        output.whitened_q.data(), output.whitened_k.data());
    if (status != Status::OK) {
        output.status = status;
        return output.status;
    }

    output.query_scales.resize(static_cast<Eigen::Index>(input.n));
    output.key_weights.resize(static_cast<Eigen::Index>(input.n));
    status = exact_decomposition_prevalidated(
        output.whitened_q.data(), output.whitened_k.data(), input.n, input.d, input.d,
        input.epsilon, input.log_clip_min, input.log_clip_max,
        output.query_scales.data(), output.key_weights.data(), &output.sigma_squared);
    if (status != Status::OK) {
        output.status = status;
        return output.status;
    }

    if (!output.metric_m.allFinite() || !output.whitening_w.allFinite() ||
        !std::isfinite(output.condition_number) || !std::isfinite(output.sigma_squared)) {
        output.status = Status::Overflow;
        return output.status;
    }
    const f32 whitening_residual = whitening_metric_residual(output);
    if (!std::isfinite(whitening_residual) || whitening_residual > 2e-4f) {
        output.status = Status::Eigendecomposition;
        return output.status;
    }

    output.status = Status::OK;
    return output.status;
}

Phase1Output phase1_forward(const Phase1Input& input) {
    Phase1Output output;
    phase1_forward_into(input, output);
    return output;
}

FrozenGateReport evaluate_frozen_gate(const Phase1Output& output) {
    FrozenGateReport report;
    report.output_status_ok = output.status == Status::OK;
    report.dimensions_valid = output.n > 0 && output.d > 0 &&
        output.metric_m.rows() == static_cast<Eigen::Index>(output.d) &&
        output.metric_m.cols() == static_cast<Eigen::Index>(output.d) &&
        output.whitening_w.rows() == static_cast<Eigen::Index>(output.d) &&
        output.whitening_w.cols() == static_cast<Eigen::Index>(output.d);
    report.condition_number = output.condition_number;
    report.finite_values = report.dimensions_valid &&
        output.whitened_q.allFinite() && output.whitened_k.allFinite() &&
        output.query_scales.allFinite() && output.key_weights.allFinite() &&
        output.metric_m.allFinite() && output.whitening_w.allFinite() &&
        std::isfinite(output.condition_number) && std::isfinite(output.sigma_squared);
    report.condition_number_valid = report.finite_values &&
        output.condition_number > 0.0f &&
        output.condition_number <= CONDITION_NUMBER_MAX_DEFAULT;

    if (report.dimensions_valid && report.finite_values) {
        Eigen::SelfAdjointEigenSolver<MatrixXf> solver(
            output.metric_m, Eigen::EigenvaluesOnly);
        if (solver.info() == Eigen::Success && solver.eigenvalues().size() > 0) {
            report.minimum_eigenvalue = solver.eigenvalues().minCoeff();
        }
    }
    report.minimum_eigenvalue_valid = std::isfinite(report.minimum_eigenvalue) &&
        report.minimum_eigenvalue >= LAMBDA_MIN_THRESHOLD;
    report.whitening_residual = whitening_metric_residual(output);
    report.whitening_isometry_valid = std::isfinite(report.whitening_residual) &&
        report.whitening_residual <= 2e-4f;
    return report;
}

bool check_frozen_gate_criteria(const Phase1Output& output) {
    return evaluate_frozen_gate(output).passed();
}

DistancePrimitiveFn get_distance_primitive(const Phase1Output& output) {
    if (output.status != Status::OK || output.metric_m.rows() != static_cast<Eigen::Index>(output.d)) {
        return nullptr;
    }
    return &distance_primitive_adapter;
}

void phase1_output_release(Phase1Output& output) {
    output.whitened_q.resize(0, 0);
    output.whitened_k.resize(0, 0);
    output.query_scales.resize(0);
    output.key_weights.resize(0);
    output.metric_m.resize(0, 0);
    output.whitening_w.resize(0, 0);
    output.n = 0;
    output.d = 0;
    output.condition_number = 0.0f;
    output.sigma_squared = 0.0f;
    output.status = Status::OK;
}

Status phase1_output_copy(const Phase1Output& src, Phase1Output& dst) {
    dst.whitened_q = src.whitened_q;
    dst.whitened_k = src.whitened_k;
    dst.query_scales = src.query_scales;
    dst.key_weights = src.key_weights;
    dst.metric_m = src.metric_m;
    dst.whitening_w = src.whitening_w;
    dst.condition_number = src.condition_number;
    dst.sigma_squared = src.sigma_squared;
    dst.n = src.n;
    dst.d = src.d;
    dst.status = src.status;
    return Status::OK;
}

} // namespace smao
