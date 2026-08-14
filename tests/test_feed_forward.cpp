#include <gtest/gtest.h>

#include "attention/feed_forward.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_ffn_config(Activation activation) {
    TransformerConfig config;
    config.vocabulary_size = 16;
    config.context_length = 32;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 1;
    config.activation = activation;
    return config;
}

void set_scalar_parameters(ParameterStore& parameters,
                           float up_weight,
                           float up_bias,
                           float down_weight,
                           float down_bias) {
    parameters.find("ffn.up.weight")->value.data()[0] = up_weight;
    parameters.find("ffn.up.bias")->value.data()[0] = up_bias;
    parameters.find("ffn.down.weight")->value.data()[0] = down_weight;
    parameters.find("ffn.down.weight")->value.data()[1] = 0.0f;
    parameters.find("ffn.down.bias")->value.data()[0] = down_bias;
    parameters.find("ffn.down.bias")->value.data()[1] = down_bias + 1.0f;
}

TEST(FeedForwardTest, RegistersStableParametersAndComputesSiLU) {
    const TransformerConfig config = make_ffn_config(Activation::SiLU);
    ParameterStore parameters;
    FeedForward feed_forward;
    std::string error;
    ASSERT_TRUE(feed_forward.register_parameters(config, parameters, &error)) << error;
    EXPECT_EQ(feed_forward.prefix(), "ffn");
    EXPECT_EQ(feed_forward.hidden_size(), 2u);
    EXPECT_EQ(feed_forward.feed_forward_size(), 1u);
    EXPECT_EQ(parameters.names(), (std::vector<std::string>{
        "ffn.down.bias", "ffn.down.weight", "ffn.up.bias", "ffn.up.weight"}));

    set_scalar_parameters(parameters, 1.0f, 0.0f, 1.0f, 0.0f);
    Tensor input;
    ASSERT_TRUE(input.reset({1, 1, 2}));
    input.data()[0] = 1.0f;
    input.data()[1] = 0.0f;
    Tensor output;
    ASSERT_TRUE(feed_forward.forward(input, parameters, output, &error)) << error;
    EXPECT_NEAR(output.data()[0], 1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
    EXPECT_FLOAT_EQ(output.data()[1], 1.0f);
}

TEST(FeedForwardTest, ComputesGELUAndPreservesTokenShape) {
    const TransformerConfig config = make_ffn_config(Activation::GELU);
    ParameterStore parameters;
    FeedForward feed_forward;
    ASSERT_TRUE(feed_forward.register_parameters(config, parameters));
    set_scalar_parameters(parameters, 1.0f, 0.0f, 1.0f, 0.0f);

    Tensor input;
    ASSERT_TRUE(input.reset({2, 3, 2}));
    input.fill(1.0f);
    Tensor output;
    ASSERT_TRUE(feed_forward.forward(input, parameters, output));
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 3, 2}));
    const float expected = 0.841191f;
    for (std::size_t token = 0; token < 6; ++token) {
        EXPECT_NEAR(output.data()[token * 2], expected, 1e-5f);
        EXPECT_FLOAT_EQ(output.data()[token * 2 + 1], 1.0f);
    }
}

TEST(FeedForwardTest, RejectsDuplicateRegistrationShapeAndNonfiniteInput) {
    const TransformerConfig config = make_ffn_config(Activation::GELU);
    ParameterStore parameters;
    FeedForward first;
    FeedForward second;
    std::string error;
    ASSERT_TRUE(first.register_parameters(config, parameters, &error)) << error;
    EXPECT_FALSE(second.register_parameters(config, parameters, &error));
    EXPECT_EQ(error, "feed-forward parameter name already exists");

    Tensor wrong_shape;
    ASSERT_TRUE(wrong_shape.reset({1, 2, 3}));
    Tensor output;
    EXPECT_FALSE(first.forward(wrong_shape, parameters, output, &error));
    EXPECT_EQ(error, "feed-forward input shape does not match hidden size");

    Tensor nonfinite;
    ASSERT_TRUE(nonfinite.reset({1, 1, 2}));
    nonfinite.fill(0.0f);
    nonfinite.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(first.forward(nonfinite, parameters, output, &error));
    EXPECT_EQ(error, "feed-forward input contains NaN or infinity");
}

} // namespace
} // namespace attention
