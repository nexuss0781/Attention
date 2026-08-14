#include <gtest/gtest.h>

#include "attention/normalization.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_norm_config() {
    TransformerConfig config;
    config.vocabulary_size = 16;
    config.context_length = 32;
    config.layer_count = 1;
    config.hidden_size = 4;
    config.attention_head_count = 2;
    config.feed_forward_size = 8;
    return config;
}

TEST(NormalizationTest, RegistersStableParametersAndNormalizesTokens) {
    ParameterStore parameters;
    Normalization normalization;
    ASSERT_TRUE(normalization.register_parameters(make_norm_config(), parameters));
    EXPECT_EQ(normalization.prefix(), "norm");
    EXPECT_EQ(normalization.hidden_size(), 4u);
    EXPECT_FLOAT_EQ(normalization.epsilon(), 1e-5f);
    EXPECT_EQ(parameters.names(), (std::vector<std::string>{"norm.bias", "norm.weight"}));

    parameters.find("norm.weight")->value.fill(1.0f);
    parameters.find("norm.bias")->value.fill(0.0f);
    Tensor input;
    ASSERT_TRUE(input.reset({1, 1, 4}));
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    input.data()[2] = 3.0f;
    input.data()[3] = 4.0f;
    Tensor output;
    std::string error;
    ASSERT_TRUE(normalization.forward(input, parameters, output, &error)) << error;
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{1, 1, 4}));
    EXPECT_NEAR(output.data()[0], -1.341635f, 1e-5f);
    EXPECT_NEAR(output.data()[1], -0.447212f, 1e-5f);
    EXPECT_NEAR(output.data()[2], 0.447212f, 1e-5f);
    EXPECT_NEAR(output.data()[3], 1.341635f, 1e-5f);
}

TEST(NormalizationTest, AppliesAffineParametersAndPreservesBatchSequenceShape) {
    ParameterStore parameters;
    Normalization normalization;
    ASSERT_TRUE(normalization.register_parameters(make_norm_config(), parameters));
    for (std::size_t i = 0; i < 4; ++i) {
        parameters.find("norm.weight")->value.data()[i] = 2.0f;
        parameters.find("norm.bias")->value.data()[i] = static_cast<float>(i);
    }
    Tensor input;
    ASSERT_TRUE(input.reset({2, 3, 4}));
    input.fill(1.0f);
    Tensor output;
    ASSERT_TRUE(normalization.forward(input, parameters, output));
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 3, 4}));
    for (std::size_t token = 0; token < 6; ++token) {
        for (std::size_t hidden = 0; hidden < 4; ++hidden) {
            EXPECT_FLOAT_EQ(output.data()[token * 4 + hidden], static_cast<float>(hidden));
        }
    }
}

TEST(NormalizationTest, RejectsDuplicateRegistrationWrongShapeAndNonfiniteInput) {
    const TransformerConfig config = make_norm_config();
    ParameterStore parameters;
    Normalization first;
    Normalization second;
    std::string error;
    ASSERT_TRUE(first.register_parameters(config, parameters, &error)) << error;
    EXPECT_FALSE(second.register_parameters(config, parameters, &error));
    EXPECT_EQ(error, "normalization parameter name already exists");

    Tensor wrong_shape;
    ASSERT_TRUE(wrong_shape.reset({1, 1, 3}));
    Tensor output;
    EXPECT_FALSE(first.forward(wrong_shape, parameters, output, &error));
    EXPECT_EQ(error, "normalization input shape does not match hidden size");

    Tensor nonfinite;
    ASSERT_TRUE(nonfinite.reset({1, 1, 4}));
    nonfinite.fill(0.0f);
    nonfinite.data()[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(first.forward(nonfinite, parameters, output, &error));
    EXPECT_EQ(error, "normalization input contains NaN or infinity");
}

} // namespace
} // namespace attention
