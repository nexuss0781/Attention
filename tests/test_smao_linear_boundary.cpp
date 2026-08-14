#include <gtest/gtest.h>

#include "attention/linear_attention.h"
#include "attention/smao_linear_boundary.h"

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

smao::Phase1Output make_valid_smao_output() {
    smao::Phase1Output output;
    output.status = smao::Status::OK;
    output.n = 2;
    output.d = 2;
    output.whitened_q.resize(2, 2);
    output.whitened_k.resize(2, 2);
    output.whitened_q << 0.0f, 0.5f, 1.0f, -0.5f;
    output.whitened_k << 0.25f, 0.0f, -0.25f, 0.75f;
    output.query_scales.resize(2);
    output.key_weights.resize(2);
    output.query_scales << 100.0f, 200.0f;
    output.key_weights << 0.01f, 0.02f;
    return output;
}

TEST(SMAOLinearBoundaryTest, AdaptsWhitenedCoordinatesWithoutConsumingScalarFactors) {
    const smao::Phase1Output smao_output = make_valid_smao_output();
    SMAOLinearBoundary boundary;
    Tensor query;
    Tensor key;
    SMAOLinearBoundaryReport report;
    ASSERT_TRUE(boundary.adapt(smao_output, query, key, &report));
    EXPECT_TRUE(report.accepted);
    EXPECT_TRUE(report.whitened_coordinates_consumed);
    EXPECT_FALSE(report.scalar_factors_consumed);
    EXPECT_EQ(report.sequence_length, 2u);
    EXPECT_EQ(report.hidden_size, 2u);
    EXPECT_EQ(query.shape(), (std::vector<std::size_t>{1, 2, 2}));
    EXPECT_EQ(key.shape(), query.shape());
    EXPECT_FLOAT_EQ(query.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(query.data()[3], -0.5f);
    EXPECT_FLOAT_EQ(key.data()[0], 0.25f);
    EXPECT_FLOAT_EQ(key.data()[3], 0.75f);
}

TEST(SMAOLinearBoundaryTest, AdaptedCoordinatesComposeWithLinearAttention) {
    const smao::Phase1Output smao_output = make_valid_smao_output();
    SMAOLinearBoundary boundary;
    Tensor query;
    Tensor key;
    ASSERT_TRUE(boundary.adapt(smao_output, query, key));

    Tensor value;
    ASSERT_TRUE(value.reset({1, 2, 2}));
    value.data()[0] = 1.0f;
    value.data()[1] = 2.0f;
    value.data()[2] = 3.0f;
    value.data()[3] = 4.0f;
    LinearCausalAttention attention;
    ASSERT_TRUE(attention.reset(2, 2));
    Tensor output;
    ASSERT_TRUE(attention.forward(query, key, value, output));
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{1, 2, 2}));
    EXPECT_TRUE(output.all_finite());
}

TEST(SMAOLinearBoundaryTest, RejectsFailedAndNonfiniteSmaoOutputs) {
    SMAOLinearBoundary boundary;
    Tensor query;
    Tensor key;
    SMAOLinearBoundaryReport report;
    smao::Phase1Output failed = make_valid_smao_output();
    failed.status = smao::Status::Overflow;
    EXPECT_FALSE(boundary.adapt(failed, query, key, &report));
    EXPECT_EQ(report.reason, "SMAO output status is not OK");

    smao::Phase1Output nonfinite = make_valid_smao_output();
    nonfinite.whitened_q(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(boundary.adapt(nonfinite, query, key, &report));
    EXPECT_EQ(report.reason, "SMAO whitened coordinates contain NaN or infinity");
}

} // namespace
} // namespace attention
