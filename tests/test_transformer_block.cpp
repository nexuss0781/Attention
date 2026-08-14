#include "attention/transformer_block.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>
#include <limits>

namespace {

attention::TransformerConfig make_config() {
    attention::TransformerConfig config;
    config.vocabulary_size = 16;
    config.context_length = 8;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 3;
    config.activation = attention::Activation::SiLU;
    return config;
}

void set_value(attention::ParameterStore& parameters, const std::string& name, float value) {
    auto* parameter = parameters.find(name);
    ASSERT_NE(parameter, nullptr) << name;
    parameter->value.fill(value);
}

void configure_identity_norm_and_constant_value_attention(attention::ParameterStore& parameters) {
    set_value(parameters, "layers.0.norm1.weight", 1.0f);
    set_value(parameters, "layers.0.norm1.bias", 0.0f);
    set_value(parameters, "layers.0.norm2.weight", 1.0f);
    set_value(parameters, "layers.0.norm2.bias", 0.0f);

    set_value(parameters, "layers.0.attention.q_proj.weight", 0.0f);
    set_value(parameters, "layers.0.attention.q_proj.bias", 0.0f);
    set_value(parameters, "layers.0.attention.k_proj.weight", 0.0f);
    set_value(parameters, "layers.0.attention.k_proj.bias", 0.0f);
    set_value(parameters, "layers.0.attention.v_proj.weight", 0.0f);
    set_value(parameters, "layers.0.attention.v_proj.bias", 1.0f);

    set_value(parameters, "layers.0.ffn.up.weight", 0.0f);
    set_value(parameters, "layers.0.ffn.up.bias", 0.0f);
    set_value(parameters, "layers.0.ffn.down.weight", 0.0f);
    set_value(parameters, "layers.0.ffn.down.bias", 0.0f);
}

} // namespace

TEST(TransformerBlockTest, RegistersLayerScopedParametersAndComposesCausally) {
    const auto config = make_config();
    attention::TransformerBlock block(0);
    attention::ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(block.register_parameters(config, parameters, &error)) << error;
    configure_identity_norm_and_constant_value_attention(parameters);

    EXPECT_TRUE(block.initialized());
    EXPECT_EQ(block.layer_index(), 0u);
    EXPECT_EQ(block.hidden_size(), 2u);
    EXPECT_EQ(block.context_length(), 8u);
    EXPECT_EQ(parameters.size(), 14u);
    EXPECT_NE(parameters.find("layers.0.norm1.weight"), nullptr);
    EXPECT_NE(parameters.find("layers.0.norm2.weight"), nullptr);
    EXPECT_NE(parameters.find("layers.0.attention.q_proj.weight"), nullptr);
    EXPECT_NE(parameters.find("layers.0.ffn.up.weight"), nullptr);

    attention::Tensor input;
    ASSERT_TRUE(input.reset({1, 3, 2}, attention::TensorDataType::F32,
                            attention::TensorDevice::CPU, &error)) << error;
    const std::vector<float> values{1.0f, 2.0f, -0.5f, 0.25f, 3.0f, 4.0f};
    for (std::size_t i = 0; i < values.size(); ++i) input.data()[i] = values[i];

    attention::Tensor output;
    ASSERT_TRUE(block.forward(input, parameters, output, &error)) << error;
    ASSERT_EQ(output.shape(), (std::vector<std::size_t>{1, 3, 2}));
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data()[i], values[i] + 1.0f) << "at " << i;
    }
}

TEST(TransformerBlockTest, IsDeterministicAndRejectsInvalidLifecycleInputs) {
    const auto config = make_config();
    attention::TransformerBlock block(0);
    attention::ParameterStore parameters;
    ASSERT_TRUE(block.register_parameters(config, parameters));
    configure_identity_norm_and_constant_value_attention(parameters);

    attention::Tensor input;
    ASSERT_TRUE(input.reset({1, 2, 2}));
    input.fill(0.5f);
    attention::Tensor first;
    attention::Tensor second;
    ASSERT_TRUE(block.forward(input, parameters, first));
    ASSERT_TRUE(block.forward(input, parameters, second));
    ASSERT_EQ(first.shape(), second.shape());
    for (std::size_t i = 0; i < first.size(); ++i) EXPECT_FLOAT_EQ(first.data()[i], second.data()[i]);

    attention::Tensor too_long;
    ASSERT_TRUE(too_long.reset({1, 9, 2}));
    std::string error;
    EXPECT_FALSE(block.forward(too_long, parameters, second, &error));
    EXPECT_NE(error.find("shape"), std::string::npos);

    attention::Tensor nonfinite;
    ASSERT_TRUE(nonfinite.reset({1, 1, 2}));
    nonfinite.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(block.forward(nonfinite, parameters, second, &error));

    attention::TransformerBlock duplicate(0);
    EXPECT_FALSE(duplicate.register_parameters(config, parameters, &error));
}

TEST(TransformerBlockTest, RejectsNoncausalConfiguration) {
    auto config = make_config();
    config.causal = false;
    attention::TransformerBlock block(0);
    attention::ParameterStore parameters;
    std::string error;
    EXPECT_FALSE(block.register_parameters(config, parameters, &error));
    EXPECT_NE(error.find("causal"), std::string::npos);
}

TEST(TransformerBlockTest, RejectsLayerOutsideConfiguration) {
    const auto config = make_config();
    attention::TransformerBlock block(1);
    attention::ParameterStore parameters;
    std::string error;
    EXPECT_FALSE(block.register_parameters(config, parameters, &error));
    EXPECT_NE(error.find("layer index"), std::string::npos);
}
