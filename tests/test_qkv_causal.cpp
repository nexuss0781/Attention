#include <gtest/gtest.h>

#include "attention/causal_mask.h"
#include "attention/qkv_projection.h"

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_projection_config() {
    TransformerConfig config;
    config.vocabulary_size = 32;
    config.context_length = 8;
    config.layer_count = 1;
    config.hidden_size = 4;
    config.attention_head_count = 2;
    config.feed_forward_size = 16;
    return config;
}

void set_identity(Parameter* parameter, float scale) {
    ASSERT_NE(parameter, nullptr);
    parameter->value.fill(0.0f);
    for (std::size_t index = 0; index < 4; ++index) {
        parameter->value.data()[index * 4 + index] = scale;
    }
}

TEST(QKVProjectionTest, RegistersStableParametersAndProjectsEachToken) {
    const TransformerConfig config = make_projection_config();
    ParameterStore parameters;
    QKVProjection projection;
    ASSERT_TRUE(projection.register_parameters(config, 0, parameters));
    EXPECT_TRUE(projection.initialized());
    EXPECT_EQ(projection.hidden_size(), 4u);
    EXPECT_EQ(projection.layer_index(), 0u);

    set_identity(parameters.find("layers.0.attention.q_proj.weight"), 1.0f);
    set_identity(parameters.find("layers.0.attention.k_proj.weight"), 2.0f);
    set_identity(parameters.find("layers.0.attention.v_proj.weight"), -1.0f);
    parameters.find("layers.0.attention.q_proj.bias")->value.fill(1.0f);
    parameters.find("layers.0.attention.k_proj.bias")->value.fill(0.0f);
    parameters.find("layers.0.attention.v_proj.bias")->value.fill(0.0f);

    Tensor input;
    ASSERT_TRUE(input.reset({1, 2, 4}));
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    input.data()[4] = 5.0f;
    input.data()[5] = 6.0f;
    input.data()[6] = 7.0f;
    input.data()[7] = 8.0f;

    QKVOutput output;
    std::string error;
    ASSERT_TRUE(projection.forward(input, parameters, output, &error)) << error;
    EXPECT_EQ(output.query.shape(), (std::vector<std::size_t>{1, 2, 4}));
    EXPECT_EQ(output.key.shape(), output.query.shape());
    EXPECT_EQ(output.value.shape(), output.query.shape());
    EXPECT_FLOAT_EQ(output.query.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(output.query.data()[3], 5.0f);
    EXPECT_FLOAT_EQ(output.query.data()[4], 6.0f);
    EXPECT_FLOAT_EQ(output.key.data()[0], 2.0f);
    EXPECT_FLOAT_EQ(output.key.data()[7], 16.0f);
    EXPECT_FLOAT_EQ(output.value.data()[0], -1.0f);
    EXPECT_FLOAT_EQ(output.value.data()[7], -8.0f);
}

TEST(QKVProjectionTest, RejectsShapeAndNonfiniteInput) {
    const TransformerConfig config = make_projection_config();
    ParameterStore parameters;
    QKVProjection projection;
    ASSERT_TRUE(projection.register_parameters(config, 1, parameters));
    ASSERT_TRUE(parameters.initialize(77));
    Tensor input;
    ASSERT_TRUE(input.reset({1, 2, 3}));
    QKVOutput output;
    std::string error;
    EXPECT_FALSE(projection.forward(input, parameters, output, &error));
    EXPECT_EQ(error, "QKV projection input shape does not match hidden size");

    ASSERT_TRUE(input.reset({1, 2, 4}));
    input.data()[2] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(projection.forward(input, parameters, output, &error));
    EXPECT_EQ(error, "QKV projection input contains NaN or infinity");
}

TEST(CausalMaskTest, StreamsCausalPredicateWithoutSequenceStorage) {
    CausalMask mask;
    std::string error;
    EXPECT_FALSE(mask.reset(0, &error));
    EXPECT_EQ(error, "context length must be positive");
    ASSERT_TRUE(mask.reset(8, &error));
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(mask.context_length(), 8u);
    EXPECT_EQ(mask.storage_bytes(), 0u);
    EXPECT_TRUE(mask.valid_sequence_length(1));
    EXPECT_TRUE(mask.valid_sequence_length(8));
    EXPECT_FALSE(mask.valid_sequence_length(9));
    EXPECT_TRUE(mask.allows(0, 0));
    EXPECT_TRUE(mask.allows(4, 0));
    EXPECT_TRUE(mask.allows(4, 4));
    EXPECT_FALSE(mask.allows(4, 5));
    EXPECT_FALSE(mask.allows(8, 0));
    EXPECT_FALSE(mask.allows(2, 8));
}

} // namespace
} // namespace attention
