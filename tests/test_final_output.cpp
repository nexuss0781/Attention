#include <gtest/gtest.h>

#include "attention/final_output.h"

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_config() {
    TransformerConfig config;
    config.vocabulary_size = 32;
    config.context_length = 64;
    config.layer_count = 2;
    config.hidden_size = 4;
    config.attention_head_count = 2;
    config.feed_forward_size = 8;
    return config;
}

TEST(FinalOutputTest, RegistersSeparateFinalNormParameters) {
    ParameterStore parameters;
    Normalization intermediate;
    FinalOutput final_output;
    ASSERT_TRUE(intermediate.register_parameters(make_config(), parameters));
    ASSERT_TRUE(final_output.register_parameters(make_config(), parameters));
    EXPECT_EQ(final_output.prefix(), "final_norm");
    EXPECT_EQ(final_output.hidden_size(), 4u);
    EXPECT_EQ(parameters.names(), (std::vector<std::string>{
        "final_norm.bias", "final_norm.weight", "norm.bias", "norm.weight"}));
}

TEST(FinalOutputTest, AppliesFinalNormalizationAndPreservesShape) {
    ParameterStore parameters;
    FinalOutput final_output;
    ASSERT_TRUE(final_output.register_parameters(make_config(), parameters));
    parameters.find("final_norm.weight")->value.fill(1.0f);
    parameters.find("final_norm.bias")->value.fill(0.0f);
    Tensor input;
    ASSERT_TRUE(input.reset({2, 3, 4}));
    for (std::size_t token = 0; token < 6; ++token) {
        for (std::size_t hidden = 0; hidden < 4; ++hidden) {
            input.data()[token * 4 + hidden] = static_cast<float>(hidden + 1);
        }
    }
    Tensor output;
    std::string error;
    ASSERT_TRUE(final_output.forward(input, parameters, output, &error)) << error;
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 3, 4}));
    EXPECT_NEAR(output.data()[0], -1.341635f, 1e-5f);
    EXPECT_NEAR(output.data()[1], -0.447212f, 1e-5f);
    EXPECT_NEAR(output.data()[2], 0.447212f, 1e-5f);
    EXPECT_NEAR(output.data()[3], 1.341635f, 1e-5f);
}

TEST(FinalOutputTest, RejectsDuplicateRegistrationAndNonfiniteInput) {
    ParameterStore parameters;
    FinalOutput first;
    FinalOutput second;
    std::string error;
    ASSERT_TRUE(first.register_parameters(make_config(), parameters, &error)) << error;
    EXPECT_FALSE(second.register_parameters(make_config(), parameters, &error));
    EXPECT_EQ(error, "normalization parameter name already exists");

    Tensor input;
    ASSERT_TRUE(input.reset({1, 1, 4}));
    input.fill(0.0f);
    input.data()[2] = std::numeric_limits<float>::quiet_NaN();
    Tensor output;
    EXPECT_FALSE(first.forward(input, parameters, output, &error));
    EXPECT_EQ(error, "normalization input contains NaN or infinity");
}

} // namespace
} // namespace attention
