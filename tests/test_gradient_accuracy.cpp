#include <gtest/gtest.h>
#include "smao_phase1/core/metric_assembly.h"

#include <cmath>
#include <vector>

namespace smao {
namespace {

f32 metric_loss(const std::vector<f32>& l, size_t d) {
    const MetricAssemblyResult result = metric_assembly(l.data(), d, 1e-6f, 1e4f);
    EXPECT_EQ(result.status, Status::OK);
    return result.metric_m.sum();
}

TEST(GradientAccuracyTest, MetricAssemblyAnalyticalMatchesFiniteDifference) {
    constexpr size_t d = 8;
    constexpr f32 delta = 1e-4f;
    std::vector<f32> l(d * d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        l[i * d + i] = 1.0f + 0.05f * static_cast<f32>(i);
        for (size_t j = 0; j < i; ++j) {
            l[i * d + j] = 0.01f * static_cast<f32>(i + j + 1);
        }
    }

    std::vector<f32> column_sums(d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            column_sums[j] += l[i * d + j];
        }
    }

    const std::vector<std::pair<size_t, size_t>> indices = {{0, 0}, {3, 0}, {5, 2}, {d - 1, d - 1}};
    for (const auto& [row, column] : indices) {
        ASSERT_LE(column, row);
        std::vector<f32> plus = l;
        std::vector<f32> minus = l;
        plus[row * d + column] += delta;
        minus[row * d + column] -= delta;
        const f32 finite_difference =
            (metric_loss(plus, d) - metric_loss(minus, d)) / (2.0f * delta);
        const f32 analytical = 2.0f * column_sums[column];
        const f32 scale = std::max(1.0f, std::abs(analytical));
        EXPECT_NEAR(finite_difference, analytical, 5e-2f * scale);
    }
}

TEST(GradientAccuracyTest, InvalidMetricInputIsRejected) {
    const std::vector<f32> invalid = {1.0f, 0.0f, std::numeric_limits<f32>::quiet_NaN(), 1.0f};
    const MetricAssemblyResult result = metric_assembly(invalid.data(), 2);
    EXPECT_EQ(result.status, Status::NaNInput);
}

} // namespace
} // namespace smao
