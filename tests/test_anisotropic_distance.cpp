#include <gtest/gtest.h>
#include "smao_phase1/core/anisotropic_distance.h"
#include "smao_phase1/core/metric_assembly.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace smao {
namespace {

TEST(AnisotropicDistanceTest, KernelConsistency) {
    constexpr size_t d = 64;
    constexpr f32 sigma_sq = std::sqrt(static_cast<f32>(d));
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diagonal(0.5f, 2.0f);

    std::vector<f32> l(d * d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            l[i * d + j] = i == j ? diagonal(gen) : dist(gen) * 0.5f;
        }
    }
    const MetricAssemblyResult result = metric_assembly(l.data(), d);
    ASSERT_EQ(result.status, Status::OK);

    std::vector<f32> q(d), k(d);
    for (size_t i = 0; i < d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
    }
    f32 relative_error = 0.0f;
    EXPECT_TRUE(verify_anisotropic_consistency(
        q.data(), k.data(), result.metric_m.data(), result.whitening_w.data(),
        sigma_sq, d, &relative_error));
    EXPECT_LT(relative_error, 1e-5f);
}

TEST(AnisotropicDistanceTest, DistributionPreservationUsesWeightedCdf) {
    constexpr size_t n = 1000;
    constexpr size_t d = 64;
    const f64 sigma_sq = std::sqrt(static_cast<f64>(d));
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));

    std::vector<f32> q(d);
    std::vector<f32> keys(n * d);
    for (f32& value : q) value = dist(gen);
    for (f32& value : keys) value = dist(gen);

    std::vector<f64> standard_logits(n);
    std::vector<f64> gaussian_logits(n);
    for (size_t j = 0; j < n; ++j) {
        f64 dot = 0.0;
        f64 key_norm = 0.0;
        f64 distance_squared = 0.0;
        for (size_t dim = 0; dim < d; ++dim) {
            const f64 key = keys[j * d + dim];
            dot += static_cast<f64>(q[dim]) * key;
            key_norm += key * key;
            const f64 delta = static_cast<f64>(q[dim]) - key;
            distance_squared += delta * delta;
        }
        const f64 query_norm = [&]() {
            f64 value = 0.0;
            for (f32 coordinate : q) value += static_cast<f64>(coordinate) * coordinate;
            return value;
        }();
        standard_logits[j] = dot / sigma_sq;
        gaussian_logits[j] = query_norm / (2.0 * sigma_sq) +
                            key_norm / (2.0 * sigma_sq) -
                            distance_squared / (2.0 * sigma_sq);
    }

    const auto normalize = [](const std::vector<f64>& logits) {
        const f64 maximum = *std::max_element(logits.begin(), logits.end());
        std::vector<f64> weights(logits.size());
        f64 sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            weights[i] = std::exp(logits[i] - maximum);
            sum += weights[i];
        }
        for (f64& weight : weights) weight /= sum;
        return weights;
    };

    const std::vector<f64> standard = normalize(standard_logits);
    const std::vector<f64> gaussian = normalize(gaussian_logits);
    f64 max_weight_error = 0.0;
    f64 standard_cdf = 0.0;
    f64 gaussian_cdf = 0.0;
    f64 max_cdf_error = 0.0;
    for (size_t i = 0; i < n; ++i) {
        max_weight_error = std::max(max_weight_error, std::abs(standard[i] - gaussian[i]));
    }

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return standard[a] < standard[b];
    });
    for (size_t index : order) {
        standard_cdf += standard[index];
        gaussian_cdf += gaussian[index];
        max_cdf_error = std::max(max_cdf_error, std::abs(standard_cdf - gaussian_cdf));
    }

    EXPECT_NEAR(standard_cdf, 1.0, 1e-12);
    EXPECT_NEAR(gaussian_cdf, 1.0, 1e-12);
    EXPECT_LT(max_weight_error, 1e-6);
    EXPECT_LT(max_cdf_error, 1e-6);
}

} // namespace
} // namespace smao
