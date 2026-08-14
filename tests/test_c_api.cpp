#include <gtest/gtest.h>
#include "smao_phase1/smao_phase1.h"

#include <cmath>
#include <limits>
#include <vector>

namespace {

smao_phase1_input_t make_input(
    std::vector<float>& q,
    std::vector<float>& k,
    std::vector<float>& v,
    std::vector<float>& l
) {
    q = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    k = {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    v = q;
    l = {1.0f, 0.0f, 0.0f, 1.0f};
    smao_phase1_input_t input{};
    input.q = q.data(); input.k = k.data(); input.v = v.data(); input.l = l.data();
    input.n = 3; input.d = 2; input.d_v = 2; input.precision = SMAO_F32;
    return input;
}

TEST(CApiTest, ForwardDistanceAndRelease) {
    std::vector<float> q, k, v, l;
    const smao_phase1_input_t input = make_input(q, k, v, l);
    smao_phase1_output_t output{};

    ASSERT_EQ(smao_validate_input(&input), SMAO_OK);
    ASSERT_EQ(smao_phase1_forward(&input, &output), SMAO_OK);
    ASSERT_NE(output.internal_handle, nullptr);
    ASSERT_NE(output.metric_m, nullptr);

    float distance_squared = -1.0f;
    ASSERT_EQ(smao_anisotropic_distance(output.internal_handle, q.data(), k.data(), input.d, &distance_squared), SMAO_OK);
    EXPECT_TRUE(std::isfinite(distance_squared));
    EXPECT_GT(distance_squared, 0.0f);

    smao_phase1_release(&output);
    EXPECT_EQ(output.internal_handle, nullptr);
    EXPECT_EQ(output.metric_m, nullptr);
    smao_phase1_release(&output);
}

TEST(CApiTest, RepeatedForwardReleasesPreviousOutput) {
    std::vector<float> q, k, v, l;
    const smao_phase1_input_t input = make_input(q, k, v, l);
    smao_phase1_output_t output{};

    ASSERT_EQ(smao_phase1_forward(&input, &output), SMAO_OK);
    ASSERT_NE(output.internal_handle, nullptr);
    ASSERT_NE(output.whitened_q, nullptr);
    float distance_squared = -1.0f;
    const size_t first_n = output.n;
    const size_t first_d = output.d;

    ASSERT_EQ(smao_phase1_forward(&input, &output), SMAO_OK);
    EXPECT_EQ(output.n, first_n);
    EXPECT_EQ(output.d, first_d);
    EXPECT_NE(output.internal_handle, nullptr);
    EXPECT_NE(output.whitened_q, nullptr);
    EXPECT_EQ(smao_anisotropic_distance(output.internal_handle, q.data(), k.data(), input.d - 1, &distance_squared),
              SMAO_ERROR_INVALID_INPUT);
    EXPECT_EQ(smao_anisotropic_distance(output.internal_handle, q.data(), k.data(), input.d, nullptr),
              SMAO_ERROR_INVALID_INPUT);

    smao_phase1_release(&output);
    EXPECT_EQ(output.internal_handle, nullptr);
}

TEST(CApiTest, InvalidInputDoesNotAllocateOrCrash) {
    std::vector<float> q, k, v, l;
    smao_phase1_input_t input = make_input(q, k, v, l);
    smao_phase1_output_t output{};
    input.q = nullptr;
    EXPECT_EQ(smao_phase1_forward(&input, &output), SMAO_ERROR_INVALID_INPUT);
    EXPECT_EQ(output.whitened_q, nullptr);
    EXPECT_EQ(output.internal_handle, nullptr);
    smao_phase1_release(&output);

    input = make_input(q, k, v, l);
    q[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(smao_validate_input(&input), SMAO_ERROR_NAN_INPUT);
    EXPECT_EQ(smao_phase1_forward(&input, &output), SMAO_ERROR_NAN_INPUT);
}

TEST(CApiTest, UnsupportedPrecisionIsRejected) {
    std::vector<float> q, k, v, l;
    smao_phase1_input_t input = make_input(q, k, v, l);
    input.precision = SMAO_F16;
    EXPECT_EQ(smao_validate_input(&input), SMAO_ERROR_INVALID_INPUT);
}

} // namespace
