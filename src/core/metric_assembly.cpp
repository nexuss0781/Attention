/*
 * @file metric_assembly.cpp
 * @brief Implementation of metric assembly and eigendecomposition.
 */

#include "smao_phase1/core/metric_assembly.h"
#include "smao_phase1/core/numerical_guards.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace smao {

namespace {

bool finite_matrix(const f32* data, size_t count) {
    if (data == nullptr) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

Status symmetric_eigendecomposition(
    f32* matrix,
    size_t d,
    f32* eigenvalues,
    f32* eigenvectors
) {
    if (matrix == nullptr || eigenvalues == nullptr || eigenvectors == nullptr || d == 0) {
        return Status::InvalidInput;
    }
    if (d > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) {
        return Status::InvalidInput;
    }
    if (!finite_matrix(matrix, d * d)) {
        return Status::NaNInput;
    }

    using ColMajorMatrix = Eigen::Matrix<f32, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    const Eigen::Index dimension = static_cast<Eigen::Index>(d);
    ColMajorMatrix symmetric(dimension, dimension);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            symmetric(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = matrix[i * d + j];
        }
    }

    // Explicitly symmetrize the input so tiny construction asymmetries cannot
    // change which triangle an eigensolver reads.
    symmetric = 0.5f * (symmetric + symmetric.transpose());
    Eigen::SelfAdjointEigenSolver<ColMajorMatrix> solver(symmetric, Eigen::ComputeEigenvectors);
    if (solver.info() != Eigen::Success) {
        return Status::Eigendecomposition;
    }

    const auto values = solver.eigenvalues();
    const auto vectors = solver.eigenvectors();
    for (size_t i = 0; i < d; ++i) {
        eigenvalues[i] = values(static_cast<Eigen::Index>(i));
        for (size_t j = 0; j < d; ++j) {
            // Store U in row-major form: element (row=j, column=i).
            eigenvectors[j * d + i] = vectors(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i));
        }
    }
    return Status::OK;
}

Status symmetric_matrix_sqrt(
    const f32* eigenvectors,
    const f32* eigenvalues,
    size_t d,
    f32* sqrt_matrix
) {
    if (eigenvectors == nullptr || eigenvalues == nullptr || sqrt_matrix == nullptr || d == 0) {
        return Status::InvalidInput;
    }

    for (size_t i = 0; i < d; ++i) {
        if (!std::isfinite(eigenvalues[i]) || eigenvalues[i] < 0.0f) {
            return Status::InvalidInput;
        }
    }
    if (!finite_matrix(eigenvectors, d * d)) {
        return Status::NaNInput;
    }

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f32 sum = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                const f32 u_ik = eigenvectors[i * d + k];
                const f32 u_jk = eigenvectors[j * d + k];
                sum += u_ik * std::sqrt(eigenvalues[k]) * u_jk;
            }
            sqrt_matrix[i * d + j] = sum;
        }
    }
    return finite_matrix(sqrt_matrix, d * d) ? Status::OK : Status::Overflow;
}

MetricAssemblyResult metric_assembly(
    const f32* l,
    size_t d,
    f32 epsilon,
    f32 condition_number_max
) {
    MetricAssemblyResult result;

    if (l == nullptr || d == 0 || !std::isfinite(epsilon) || epsilon < 0.0f ||
        !std::isfinite(condition_number_max) || condition_number_max <= 1.0f) {
        result.status = Status::InvalidInput;
        return result;
    }
    if (!finite_matrix(l, d * d)) {
        result.status = Status::NaNInput;
        return result;
    }

    result.metric_m.resize(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));
    result.whitening_w.resize(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));

    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f32 sum = 0.0f;
            const size_t k_max = std::min(i, j);
            for (size_t k = 0; k <= k_max; ++k) {
                sum += l[i * d + k] * l[j * d + k];
            }
            result.metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = sum;
        }
        result.metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) += epsilon;
    }

    std::vector<f32> matrix_copy(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            matrix_copy[i * d + j] = result.metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    std::vector<f32> eigenvalues(d);
    std::vector<f32> eigenvectors(d * d);
    Status eig_status = symmetric_eigendecomposition(
        matrix_copy.data(), d, eigenvalues.data(), eigenvectors.data());
    if (eig_status != Status::OK) {
        result.status = eig_status;
        return result;
    }

    result.condition_number = enforce_condition_number_bound(
        eigenvalues.data(), d, condition_number_max);
    if (!std::isfinite(result.condition_number) ||
        result.condition_number > condition_number_max) {
        result.status = Status::ConditionNumber;
        return result;
    }
    if (!validate_condition_number(eigenvalues.data(), d, condition_number_max)) {
        result.status = Status::ConditionNumber;
        return result;
    }

    // Reconstruct the returned metric from the projected eigenpairs. This is
    // essential: W and metric_m must describe the same quadratic form after
    // conditioning has modified the spectrum.
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            f32 sum = 0.0f;
            for (size_t k = 0; k < d; ++k) {
                sum += eigenvectors[i * d + k] * eigenvalues[k] * eigenvectors[j * d + k];
            }
            result.metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = sum;
        }
    }
    result.metric_m = 0.5f * (result.metric_m + result.metric_m.transpose());

    Status sqrt_status = symmetric_matrix_sqrt(
        eigenvectors.data(), eigenvalues.data(), d, result.whitening_w.data());
    if (sqrt_status != Status::OK) {
        result.status = sqrt_status;
        return result;
    }

    // Keep the two returned representations numerically identical in float32:
    // the metric exposed to callers is the quadratic form represented by the
    // final whitening matrix, not a separately rounded reconstruction.
    result.metric_m = result.whitening_w * result.whitening_w;
    result.metric_m = 0.5f * (result.metric_m + result.metric_m.transpose());
    result.status = validate_spd(result.metric_m.data(), d, 0.0f);
    return result;
}

Status validate_spd(const f32* metric, size_t d, f32 epsilon) {
    if (metric == nullptr || d == 0 || !std::isfinite(epsilon) || epsilon < 0.0f) {
        return Status::InvalidInput;
    }
    if (!finite_matrix(metric, d * d)) {
        return Status::NaNInput;
    }

    const f32 symmetry_tolerance = std::max(epsilon, 1e-5f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = i + 1; j < d; ++j) {
            if (std::abs(metric[i * d + j] - metric[j * d + i]) > symmetry_tolerance) {
                return Status::InvalidInput;
            }
        }
    }

    std::vector<f32> eigenvalues(d);
    std::vector<f32> eigenvectors(d * d);
    std::vector<f32> matrix_copy(metric, metric + d * d);
    Status status = symmetric_eigendecomposition(
        matrix_copy.data(), d, eigenvalues.data(), eigenvectors.data());
    if (status != Status::OK) {
        return status;
    }
    for (f32 value : eigenvalues) {
        if (value < epsilon || !std::isfinite(value)) {
            return Status::MetricSingular;
        }
    }
    return Status::OK;
}

} // namespace smao
