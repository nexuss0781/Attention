#include "attention/smao_linear_boundary.h"

#include <cmath>
#include <limits>

namespace attention {
namespace {

void fail(SMAOLinearBoundaryReport* report,
          const char* reason,
          std::size_t sequence_length = 0,
          std::size_t hidden_size = 0) {
    if (report == nullptr) return;
    report->accepted = false;
    report->whitened_coordinates_consumed = false;
    report->scalar_factors_consumed = false;
    report->sequence_length = sequence_length;
    report->hidden_size = hidden_size;
    report->reason = reason;
}

} // namespace

bool SMAOLinearBoundary::adapt(const smao::Phase1Output& smao_output,
                               Tensor& query,
                               Tensor& key,
                               SMAOLinearBoundaryReport* report) const {
    if (report != nullptr) *report = SMAOLinearBoundaryReport{};
    if (smao_output.status != smao::Status::OK) {
        fail(report, "SMAO output status is not OK");
        return false;
    }
    if (smao_output.n == 0 || smao_output.d == 0) {
        fail(report, "SMAO output dimensions must be positive", smao_output.n, smao_output.d);
        return false;
    }
    if (smao_output.n > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) ||
        smao_output.d > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
        fail(report, "SMAO output dimensions exceed Eigen index range", smao_output.n, smao_output.d);
        return false;
    }
    const auto expected_rows = static_cast<Eigen::Index>(smao_output.n);
    const auto expected_columns = static_cast<Eigen::Index>(smao_output.d);
    if (smao_output.whitened_q.rows() != expected_rows ||
        smao_output.whitened_q.cols() != expected_columns ||
        smao_output.whitened_k.rows() != expected_rows ||
        smao_output.whitened_k.cols() != expected_columns) {
        fail(report, "SMAO whitened coordinate shapes do not match output dimensions",
             smao_output.n, smao_output.d);
        return false;
    }
    if (!smao_output.whitened_q.allFinite() || !smao_output.whitened_k.allFinite()) {
        fail(report, "SMAO whitened coordinates contain NaN or infinity",
             smao_output.n, smao_output.d);
        return false;
    }
    if (smao_output.query_scales.size() != expected_rows ||
        smao_output.key_weights.size() != expected_rows ||
        !smao_output.query_scales.allFinite() || !smao_output.key_weights.allFinite()) {
        fail(report, "SMAO scalar decomposition factors are missing or nonfinite",
             smao_output.n, smao_output.d);
        return false;
    }
    if (!query.reset({1, smao_output.n, smao_output.d}) ||
        !key.reset({1, smao_output.n, smao_output.d})) {
        fail(report, "linear boundary tensor allocation failed", smao_output.n, smao_output.d);
        return false;
    }

    const std::size_t element_count = smao_output.n * smao_output.d;
    for (std::size_t index = 0; index < element_count; ++index) {
        query.data()[index] = smao_output.whitened_q.data()[index];
        key.data()[index] = smao_output.whitened_k.data()[index];
    }
    if (report != nullptr) {
        report->accepted = true;
        report->whitened_coordinates_consumed = true;
        report->scalar_factors_consumed = false;
        report->sequence_length = smao_output.n;
        report->hidden_size = smao_output.d;
        report->reason = "finite SMAO-whitened coordinates adapted; scalar factors intentionally not consumed";
    }
    return true;
}

} // namespace attention
