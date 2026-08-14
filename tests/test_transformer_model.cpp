#include "attention/transformer_model.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

TEST(TransformerModelTest, RunsCompleteForwardAndCausalLossDeterministically) {
    const TransformerConfig config = make_config();
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(config, parameters, &error)) << error;
    ASSERT_TRUE(model.initialized());
    EXPECT_EQ(model.vocabulary_size(), 3u);
    EXPECT_EQ(model.context_length(), 4u);
    EXPECT_EQ(model.hidden_size(), 2u);
    EXPECT_EQ(model.layer_count(), 1u);

    const std::vector<std::size_t> tokens{0, 1, 2};
    Tensor first;
    Tensor second;
    ASSERT_TRUE(model.forward(tokens, 1, 3, parameters, first, &error)) << error;
    ASSERT_TRUE(model.forward(tokens, 1, 3, parameters, second, &error)) << error;
    ASSERT_EQ(first.shape(), (std::vector<std::size_t>{1, 3, 3}));
    ASSERT_EQ(first.shape(), second.shape());
    for (std::size_t index = 0; index < first.size(); ++index) {
        EXPECT_TRUE(std::isfinite(first.data()[index]));
        EXPECT_FLOAT_EQ(first.data()[index], second.data()[index]);
        EXPECT_FLOAT_EQ(first.data()[index], 0.0f);
    }

    float loss = 0.0f;
    ASSERT_TRUE(model.causal_loss(tokens, 1, 3, parameters, loss, &error)) << error;
    EXPECT_NEAR(loss, std::log(3.0f), 1e-6f);
}

TEST(TransformerModelTest, ComputesStableCausalCrossEntropyFromFullLogits) {
    Tensor logits;
    ASSERT_TRUE(logits.reset({1, 3, 3}));
    const std::vector<float> values{1.0f, 2.0f, 3.0f,
                                    0.0f, 5.0f, 0.0f,
                                    4.0f, 1.0f, 2.0f};
    for (std::size_t index = 0; index < values.size(); ++index) logits.data()[index] = values[index];
    const std::vector<std::size_t> targets{0, 2, 1};
    float loss = 0.0f;
    std::string error;
    ASSERT_TRUE(TransformerModel::causal_cross_entropy(logits, targets, loss, &error)) << error;
    const double first = std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0)) - 3.0;
    const double second = std::log(std::exp(0.0) + std::exp(5.0) + std::exp(0.0)) - 5.0;
    EXPECT_NEAR(loss, static_cast<float>((first + second) / 2.0), 1e-6f);
}

TEST(TransformerModelTest, RejectsInvalidTokenShapesTargetsAndNonfiniteLogits) {
    const TransformerConfig config = make_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    Tensor logits;
    std::string error;
    EXPECT_FALSE(model.forward({0, 1}, 1, 3, parameters, logits, &error));
    EXPECT_FALSE(model.forward({0, 1, 3}, 1, 3, parameters, logits, &error));
    float loss = 0.0f;
    EXPECT_FALSE(TransformerModel::causal_cross_entropy(logits, {0, 1, 2}, loss, &error));

    Tensor invalid;
    ASSERT_TRUE(invalid.reset({1, 2, 3}));
    invalid.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(TransformerModel::causal_cross_entropy(invalid, {0, 1}, loss, &error));

    Tensor valid;
    ASSERT_TRUE(valid.reset({1, 2, 3}));
    EXPECT_FALSE(TransformerModel::causal_cross_entropy(valid, {0, 3}, loss, &error));
    EXPECT_FALSE(TransformerModel::causal_cross_entropy(valid, {0}, loss, &error));
}

} // namespace
} // namespace attention

namespace attention {
namespace {

TEST(TransformerModelTest, BackwardPopulatesFiniteGradientsAndMatchesCentralDifference) {
    const TransformerConfig config = make_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(42));
    const std::vector<std::size_t> tokens{0, 1, 2};
    const float step = 1e-3f;
    const Parameter* before = parameters.find("lm_head.bias");
    ASSERT_NE(before, nullptr);
    const float original_bias = before->value.data()[0];

    std::string error;
    ASSERT_TRUE(model.backward(tokens, 1, 3, parameters, step, &error)) << error;
    const Parameter* bias = parameters.find("lm_head.bias");
    ASSERT_NE(bias, nullptr);
    EXPECT_FLOAT_EQ(bias->value.data()[0], original_bias);
    bool saw_nonzero = false;
    for (const Parameter& parameter : parameters.parameters()) {
        ASSERT_TRUE(parameter.gradient.all_finite()) << parameter.name;
        for (std::size_t index = 0; index < parameter.gradient.size(); ++index) {
            saw_nonzero = saw_nonzero || std::abs(parameter.gradient.data()[index]) > 1e-7f;
        }
    }
    EXPECT_TRUE(saw_nonzero);

    ParameterStore plus = parameters;
    ParameterStore minus = parameters;
    plus.find("lm_head.bias")->value.data()[0] += step;
    minus.find("lm_head.bias")->value.data()[0] -= step;
    float plus_loss = 0.0f;
    float minus_loss = 0.0f;
    ASSERT_TRUE(model.causal_loss(tokens, 1, 3, plus, plus_loss, &error)) << error;
    ASSERT_TRUE(model.causal_loss(tokens, 1, 3, minus, minus_loss, &error)) << error;
    const float expected = (plus_loss - minus_loss) / (2.0f * step);
    EXPECT_NEAR(bias->gradient.data()[0], expected, 2e-4f);

    ParameterStore repeat = parameters;
    ASSERT_TRUE(model.backward(tokens, 1, 3, repeat, step, &error)) << error;
    ASSERT_EQ(parameters.size(), repeat.size());
    for (const std::string& name : parameters.names()) {
        const Parameter* left = parameters.find(name);
        const Parameter* right = repeat.find(name);
        ASSERT_NE(left, nullptr);
        ASSERT_NE(right, nullptr);
        ASSERT_EQ(left->gradient.size(), right->gradient.size());
        for (std::size_t index = 0; index < left->gradient.size(); ++index) {
            EXPECT_FLOAT_EQ(left->gradient.data()[index], right->gradient.data()[index]);
        }
    }
}

TEST(TransformerModelTest, BackwardRejectsInvalidDifferenceStepAndTargets) {
    const TransformerConfig config = make_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    const std::vector<std::size_t> tokens{0, 1, 2};
    std::string error;
    EXPECT_FALSE(model.backward(tokens, 1, 3, parameters, 0.0f, &error));
    EXPECT_NE(error.find("difference step"), std::string::npos);
    EXPECT_FALSE(model.backward({0, 1, 9}, 1, 3, parameters, 1e-3f, &error));
    EXPECT_NE(error.find("vocabulary"), std::string::npos);
    EXPECT_FALSE(model.backward({0}, 1, 1, parameters, 1e-3f, &error));
    EXPECT_NE(error.find("sequence"), std::string::npos);
}

} // namespace
} // namespace attention
