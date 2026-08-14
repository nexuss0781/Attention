/*
 * @file whiten_coordinates.cpp
 * @brief Implementation of coordinate whitening.
 */

#include "smao_phase1/core/whiten_coordinates.h"
#include "smao_phase1/core/numerical_guards.h"

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(SMAO_HAS_CBLAS)
#include <cblas.h>
#endif

namespace smao {

namespace {

bool finite_buffer(const f32* data, size_t count) {
    if (data == nullptr && count != 0) return false;
    if (count == 0) return true;
    if (count > static_cast<size_t>(std::numeric_limits<Eigen::Index>::max())) return false;
    return Eigen::Map<const VectorXf>(data, static_cast<Eigen::Index>(count)).allFinite();
}

bool valid_dimensions(size_t n, size_t d) {
    return n != 0 && d != 0 &&
           n <= static_cast<size_t>(std::numeric_limits<Eigen::Index>::max()) &&
           d <= static_cast<size_t>(std::numeric_limits<Eigen::Index>::max());
}

void whiten_row(const f32* input, const f32* whitening_w, size_t d, f32* output) {
    std::fill(output, output + d, 0.0f);
    for (size_t k = 0; k < d; ++k) {
        const f32 value = input[k];
        const f32* weight_row = whitening_w + k * d;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (size_t j = 0; j < d; ++j) {
            output[j] += value * weight_row[j];
        }
    }
}

#if defined(__AVX512F__)
inline void whiten_pair_row_avx512(
    const f32* q_input,
    const f32* k_input,
    const f32* whitening_w,
    size_t d,
    f32* q_output,
    f32* k_output
) {
    __m512 q_acc[8];
    __m512 k_acc[8];
    const size_t blocks = d / 16;
    for (size_t block = 0; block < blocks; ++block) {
        q_acc[block] = _mm512_setzero_ps();
        k_acc[block] = _mm512_setzero_ps();
    }
    for (size_t k = 0; k < d; ++k) {
        const __m512 q_value = _mm512_set1_ps(q_input[k]);
        const __m512 k_value = _mm512_set1_ps(k_input[k]);
        const f32* weight_row = whitening_w + k * d;
        for (size_t block = 0; block < blocks; ++block) {
            const __m512 weights = _mm512_loadu_ps(weight_row + block * 16);
            q_acc[block] = _mm512_fmadd_ps(q_value, weights, q_acc[block]);
            k_acc[block] = _mm512_fmadd_ps(k_value, weights, k_acc[block]);
        }
    }
    for (size_t block = 0; block < blocks; ++block) {
        _mm512_storeu_ps(q_output + block * 16, q_acc[block]);
        _mm512_storeu_ps(k_output + block * 16, k_acc[block]);
    }
}
#endif

} // namespace

Status whiten_coordinates_pair_prevalidated(
    const f32* q,
    const f32* k,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* q_whitened,
    f32* k_whitened
) {
    if (q == nullptr || k == nullptr || whitening_w == nullptr ||
        q_whitened == nullptr || k_whitened == nullptr || !valid_dimensions(n, d)) {
        return Status::InvalidInput;
    }

#if defined(__AVX512F__)
    std::atomic<bool> input_invalid{false};
    std::atomic<bool> output_overflow{false};
#endif
    if (d <= 128) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long row = 0; row < static_cast<long long>(n); ++row) {
            const size_t offset = static_cast<size_t>(row) * d;
            f32* q_output = q_whitened + offset;
            f32* k_output = k_whitened + offset;
            const f32* q_input = q + offset;
            const f32* k_input = k + offset;
#if defined(__AVX512F__)
            if ((d % 16) == 0) {
                bool row_input_invalid = false;
                const __m512 maximum = _mm512_set1_ps(std::numeric_limits<f32>::max());
                for (size_t block = 0; block < d; block += 16) {
                    const __m512 q_values = _mm512_loadu_ps(q_input + block);
                    const __m512 k_values = _mm512_loadu_ps(k_input + block);
                    const __mmask16 q_nan = _mm512_cmp_ps_mask(q_values, q_values, _CMP_UNORD_Q);
                    const __mmask16 k_nan = _mm512_cmp_ps_mask(k_values, k_values, _CMP_UNORD_Q);
                    const __mmask16 q_inf = _mm512_cmp_ps_mask(_mm512_abs_ps(q_values), maximum, _CMP_GT_OQ);
                    const __mmask16 k_inf = _mm512_cmp_ps_mask(_mm512_abs_ps(k_values), maximum, _CMP_GT_OQ);
                    row_input_invalid = row_input_invalid || q_nan != 0 || k_nan != 0 || q_inf != 0 || k_inf != 0;
                }
                if (row_input_invalid) {
                    input_invalid.store(true, std::memory_order_relaxed);
                    continue;
                }
                whiten_pair_row_avx512(q_input, k_input, whitening_w, d, q_output, k_output);
                for (size_t block = 0; block < d; block += 16) {
                    const __m512 q_values = _mm512_loadu_ps(q_output + block);
                    const __m512 k_values = _mm512_loadu_ps(k_output + block);
                    const __mmask16 q_bad = _mm512_cmp_ps_mask(q_values, q_values, _CMP_UNORD_Q) |
                        _mm512_cmp_ps_mask(_mm512_abs_ps(q_values), maximum, _CMP_GT_OQ);
                    const __mmask16 k_bad = _mm512_cmp_ps_mask(k_values, k_values, _CMP_UNORD_Q) |
                        _mm512_cmp_ps_mask(_mm512_abs_ps(k_values), maximum, _CMP_GT_OQ);
                    if (q_bad != 0 || k_bad != 0) output_overflow.store(true, std::memory_order_relaxed);
                }
                continue;
            }
#endif
            std::fill(q_output, q_output + d, 0.0f);
            std::fill(k_output, k_output + d, 0.0f);
            for (size_t dimension = 0; dimension < d; ++dimension) {
                const f32 q_value = q_input[dimension];
                const f32 k_value = k_input[dimension];
                const f32* weight_row = whitening_w + dimension * d;
#ifdef _OPENMP
#pragma omp simd
#endif
                for (size_t j = 0; j < d; ++j) {
                    q_output[j] += q_value * weight_row[j];
                    k_output[j] += k_value * weight_row[j];
                }
            }
        }
    } else {
        using RowMajorMap = Eigen::Map<const MatrixXf>;
        using MutableRowMajorMap = Eigen::Map<MatrixXf>;
        const Eigen::Index rows = static_cast<Eigen::Index>(n);
        const Eigen::Index cols = static_cast<Eigen::Index>(d);
        RowMajorMap q_map(q, rows, cols);
        RowMajorMap k_map(k, rows, cols);
        RowMajorMap w_map(whitening_w, cols, cols);
        MutableRowMajorMap q_output(q_whitened, rows, cols);
        MutableRowMajorMap k_output(k_whitened, rows, cols);
#if defined(SMAO_HAS_CBLAS)
        if (n <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            d <= static_cast<size_t>(std::numeric_limits<int>::max())) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(d), static_cast<int>(d),
                1.0f, q, static_cast<int>(d), whitening_w, static_cast<int>(d),
                0.0f, q_whitened, static_cast<int>(d));
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(d), static_cast<int>(d),
                1.0f, k, static_cast<int>(d), whitening_w, static_cast<int>(d),
                0.0f, k_whitened, static_cast<int>(d));
        } else {
            q_output.noalias() = q_map * w_map;
            k_output.noalias() = k_map * w_map;
        }
#else
        q_output.noalias() = q_map * w_map;
        k_output.noalias() = k_map * w_map;
#endif
    }
#if defined(__AVX512F__)
    if (d <= 128 && (d % 16) == 0) {
        if (input_invalid.load(std::memory_order_relaxed)) return Status::NaNInput;
        if (output_overflow.load(std::memory_order_relaxed)) return Status::Overflow;
        return Status::OK;
    }
#endif
    const Eigen::Map<const VectorXf> q_values(q_whitened, static_cast<Eigen::Index>(n * d));
    const Eigen::Map<const VectorXf> k_values(k_whitened, static_cast<Eigen::Index>(n * d));
    if (q_values.allFinite() && k_values.allFinite()) return Status::OK;
    if (q_values.array().isNaN().any() || k_values.array().isNaN().any()) return Status::NaNInput;
    return Status::Overflow;
}

Status whiten_coordinates_prevalidated(
    const f32* x,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* x_whitened
) {
    if (x == nullptr || whitening_w == nullptr || x_whitened == nullptr ||
        !valid_dimensions(n, d)) return Status::InvalidInput;
    if (d <= 128) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long row = 0; row < static_cast<long long>(n); ++row) {
            whiten_row(x + static_cast<size_t>(row) * d, whitening_w, d,
                       x_whitened + static_cast<size_t>(row) * d);
        }
    } else {
        using RowMajorMap = Eigen::Map<const MatrixXf>;
        using MutableRowMajorMap = Eigen::Map<MatrixXf>;
        const Eigen::Index rows = static_cast<Eigen::Index>(n);
        const Eigen::Index cols = static_cast<Eigen::Index>(d);
        RowMajorMap x_map(x, rows, cols);
        RowMajorMap w_map(whitening_w, cols, cols);
        MutableRowMajorMap output_map(x_whitened, rows, cols);
#if defined(SMAO_HAS_CBLAS)
        if (n <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            d <= static_cast<size_t>(std::numeric_limits<int>::max())) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(n), static_cast<int>(d), static_cast<int>(d),
                1.0f, x, static_cast<int>(d), whitening_w, static_cast<int>(d),
                0.0f, x_whitened, static_cast<int>(d));
        } else {
            output_map.noalias() = x_map * w_map;
        }
#else
        output_map.noalias() = x_map * w_map;
#endif
    }
    return finite_buffer(x_whitened, n * d) ? Status::OK : Status::Overflow;
}

Status whiten_coordinates(
    const f32* x,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* x_whitened
) {
    if (x == nullptr || whitening_w == nullptr || x_whitened == nullptr ||
        !valid_dimensions(n, d)) return Status::InvalidInput;
    if (!finite_buffer(x, n * d) || !finite_buffer(whitening_w, d * d)) {
        return Status::NaNInput;
    }
    return whiten_coordinates_prevalidated(x, whitening_w, n, d, x_whitened);
}

MatrixXf whiten_coordinates_eigen(const MatrixXf& x, const MatrixXf& whitening_w) {
    if (x.cols() != whitening_w.rows()) return MatrixXf();
    return x * whitening_w;
}

Status whiten_coordinates_batch(
    const f32* q,
    const f32* k,
    const f32* whitening_w,
    size_t n,
    size_t d,
    f32* q_whitened,
    f32* k_whitened
) {
    if (q == nullptr || k == nullptr || whitening_w == nullptr ||
        q_whitened == nullptr || k_whitened == nullptr || !valid_dimensions(n, d)) {
        return Status::InvalidInput;
    }
    if (!finite_buffer(q, n * d) || !finite_buffer(k, n * d) ||
        !finite_buffer(whitening_w, d * d)) return Status::NaNInput;
    return whiten_coordinates_pair_prevalidated(
        q, k, whitening_w, n, d, q_whitened, k_whitened);
}

bool verify_whitening_isometry(
    const MatrixXf& whitening_w,
    const MatrixXf& metric_m,
    size_t d,
    size_t num_samples,
    f32 max_relative_error
) {
    if (d == 0 || num_samples == 0 || whitening_w.rows() != static_cast<Eigen::Index>(d) ||
        whitening_w.cols() != static_cast<Eigen::Index>(d) ||
        metric_m.rows() != static_cast<Eigen::Index>(d) ||
        metric_m.cols() != static_cast<Eigen::Index>(d) ||
        !std::isfinite(max_relative_error) || max_relative_error < 0.0f ||
        !whitening_w.allFinite() || !metric_m.allFinite()) return false;

    for (size_t sample = 0; sample < num_samples; ++sample) {
        VectorXf x(static_cast<Eigen::Index>(d));
        for (size_t i = 0; i < d; ++i) {
            x(static_cast<Eigen::Index>(i)) = static_cast<f32>(
                std::sin(static_cast<f64>(sample) * 1000.0 + static_cast<f64>(i) * 0.5) * 0.5 + 0.5);
        }
        const f32 norm = x.norm();
        if (!(norm > 0.0f) || !std::isfinite(norm)) return false;
        x /= norm;
        f64 wx_norm_sq = 0.0;
        f64 x_m_x = 0.0;
        for (size_t i = 0; i < d; ++i) {
            f64 transformed = 0.0;
            for (size_t j = 0; j < d; ++j) {
                transformed += static_cast<f64>(whitening_w(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))) * x(static_cast<Eigen::Index>(j));
            }
            wx_norm_sq += transformed * transformed;
        }
        for (size_t i = 0; i < d; ++i) {
            f64 metric_row = 0.0;
            for (size_t j = 0; j < d; ++j) {
                metric_row += static_cast<f64>(metric_m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))) * x(static_cast<Eigen::Index>(j));
            }
            x_m_x += static_cast<f64>(x(static_cast<Eigen::Index>(i))) * metric_row;
        }
        if (!std::isfinite(wx_norm_sq) || !std::isfinite(x_m_x)) return false;
        const f64 diff = std::abs(wx_norm_sq - x_m_x);
        const f64 relative_error = x_m_x > 1e-10 ? diff / x_m_x : diff;
        if (relative_error > max_relative_error) return false;
    }
    return true;
}

} // namespace smao
