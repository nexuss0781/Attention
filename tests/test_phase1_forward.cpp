#include <gtest/gtest.h>
#include "smao_phase1/core/phase1_forward.h"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace smao {
namespace {

std::vector<f32> generate_random_l(size_t d, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    std::uniform_real_distribution<f32> diag_dist(0.5f, 1.5f);
    std::vector<f32> l(d * d, 0.0f);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            l[i * d + j] = i == j ? diag_dist(gen) : dist(gen) * 0.3f;
        }
    }
    return l;
}

Phase1Input make_valid_input(
    std::vector<f32>& q,
    std::vector<f32>& k,
    std::vector<f32>& v,
    std::vector<f32>& l,
    size_t n,
    size_t d
) {
    std::mt19937 gen(42);
    std::normal_distribution<f32> dist(0.0f, 1.0f / std::sqrt(static_cast<f32>(d)));
    q.resize(n * d);
    k.resize(n * d);
    v.resize(n * d);
    for (size_t i = 0; i < n * d; ++i) {
        q[i] = dist(gen);
        k[i] = dist(gen);
        v[i] = dist(gen);
    }
    l = generate_random_l(d);
    Phase1Input input;
    input.q = q.data(); input.k = k.data(); input.v = v.data(); input.l = l.data();
    input.n = n; input.d = d; input.d_v = d; input.precision = Precision::F32;
    return input;
}

TEST(Phase1ForwardTest, CompletePipeline) {
    constexpr size_t n = 256;
    constexpr size_t d = 64;
    std::vector<f32> q, k, v, l;
    const Phase1Input input = make_valid_input(q, k, v, l, n, d);
    const Phase1Output output = phase1_forward(input);

    ASSERT_EQ(output.status, Status::OK);
    EXPECT_EQ(output.n, n);
    EXPECT_EQ(output.d, d);
    EXPECT_EQ(output.whitened_q.rows(), static_cast<Eigen::Index>(n));
    EXPECT_EQ(output.whitened_q.cols(), static_cast<Eigen::Index>(d));
    EXPECT_EQ(output.whitened_k.rows(), static_cast<Eigen::Index>(n));
    EXPECT_EQ(output.whitened_k.cols(), static_cast<Eigen::Index>(d));
    EXPECT_EQ(output.query_scales.size(), static_cast<Eigen::Index>(n));
    EXPECT_EQ(output.key_weights.size(), static_cast<Eigen::Index>(n));
    EXPECT_EQ(output.metric_m.rows(), static_cast<Eigen::Index>(d));
    EXPECT_EQ(output.whitening_w.rows(), static_cast<Eigen::Index>(d));
    EXPECT_GT(output.condition_number, 0.0f);
    EXPECT_LE(output.condition_number, 1e4f);
    EXPECT_FLOAT_EQ(output.sigma_squared, std::sqrt(static_cast<f32>(d)));
    EXPECT_TRUE(output.whitened_q.allFinite());
    EXPECT_TRUE(output.whitened_k.allFinite());
    EXPECT_TRUE(output.query_scales.allFinite());
    EXPECT_TRUE(output.key_weights.allFinite());
    EXPECT_TRUE(output.metric_m.allFinite());
    EXPECT_TRUE(output.whitening_w.allFinite());
    for (Eigen::Index i = 0; i < output.query_scales.size(); ++i) {
        EXPECT_GT(output.query_scales(i), 0.0f);
        EXPECT_GT(output.key_weights(i), 0.0f);
    }
    EXPECT_TRUE(check_frozen_gate_criteria(output));
}

TEST(Phase1ForwardTest, FrozenGateCriteria) {
    constexpr size_t n = 128;
    constexpr size_t d = 32;
    std::vector<f32> q, k, v, l;
    const Phase1Input input = make_valid_input(q, k, v, l, n, d);
    const Phase1Output output = phase1_forward(input);
    ASSERT_EQ(output.status, Status::OK);
    EXPECT_TRUE(check_frozen_gate_criteria(output));
    ASSERT_NE(get_distance_primitive(output), nullptr);
    const f32 distance = get_distance_primitive(output)(q.data(), k.data(), output.metric_m, d);
    EXPECT_TRUE(std::isfinite(distance));
    EXPECT_GE(distance, 0.0f);
}

TEST(Phase1ForwardTest, InvalidInputs) {
    constexpr size_t n = 4;
    constexpr size_t d = 2;
    std::vector<f32> q(n * d, 0.0f), k(n * d, 0.0f), v(n * d, 0.0f);
    std::vector<f32> l = {1.0f, 0.0f, 0.0f, 1.0f};
    Phase1Input input = make_valid_input(q, k, v, l, n, d);

    input.q = nullptr;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    input.k = nullptr;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    input.v = nullptr;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    input.l = nullptr;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    input.n = 0;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    q[0] = std::numeric_limits<f32>::quiet_NaN();
    EXPECT_EQ(phase1_forward(input).status, Status::NaNInput);
    input = make_valid_input(q, k, v, l, n, d);
    l[1] = 1.0f;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
    input = make_valid_input(q, k, v, l, n, d);
    input.precision = Precision::F16;
    EXPECT_EQ(phase1_forward(input).status, Status::InvalidInput);
}

} // namespace
} // namespace smao
