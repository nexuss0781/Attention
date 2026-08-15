#include "attention/validation.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig validation_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

TEST(ValidationEvaluatorTest, ComputesWeightedFiniteLossAndResetsLoader) {
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(validation_config(), parameters, &error)) << error;
    ASSERT_TRUE(parameters.initialize(41, &error)) << error;
    const std::vector<std::size_t> tokens{0, 1, 2, 0, 1, 2, 0, 1};
    TrainingBatchLoader loader;
    ASSERT_TRUE(loader.initialize(tokens, 1, 4, true, &error)) << error;

    float first_loss = 0.0f;
    float second_loss = 0.0f;
    ASSERT_TRUE(model.causal_loss({0, 1, 2, 0}, 1, 4, parameters, first_loss, &error)) << error;
    ASSERT_TRUE(model.causal_loss({1, 2, 0, 1}, 1, 4, parameters, second_loss, &error)) << error;
    const std::vector<std::string> names = parameters.names();
    std::vector<std::vector<float>> before;
    for (const std::string& name : names) {
        const Parameter* parameter = parameters.find(name);
        before.emplace_back(parameter->value.data(), parameter->value.data() + parameter->value.size());
    }

    ValidationResult result;
    ASSERT_TRUE(ValidationEvaluator::evaluate(model, loader, parameters, result, &error)) << error;
    EXPECT_EQ(result.batches, 2U);
    EXPECT_EQ(result.prediction_tokens, 6U);
    EXPECT_TRUE(std::isfinite(result.mean_loss));
    EXPECT_FLOAT_EQ(result.mean_loss, (first_loss + second_loss) / 2.0f);
    EXPECT_EQ(loader.batches_emitted(), 0U);

    ValidationResult repeated;
    ASSERT_TRUE(ValidationEvaluator::evaluate(model, loader, parameters, repeated, &error)) << error;
    EXPECT_FLOAT_EQ(repeated.mean_loss, result.mean_loss);
    for (std::size_t parameter_index = 0; parameter_index < names.size(); ++parameter_index) {
        const Parameter* parameter = parameters.find(names[parameter_index]);
        ASSERT_NE(parameter, nullptr);
        EXPECT_EQ(std::vector<float>(parameter->value.data(), parameter->value.data() + parameter->value.size()),
                  before[parameter_index]);
    }
}

TEST(ValidationEvaluatorTest, RejectsEmptyAndNonfiniteInputs) {
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(validation_config(), parameters, &error)) << error;
    ASSERT_TRUE(parameters.initialize(43, &error)) << error;
    TrainingBatchLoader empty_loader;
    ASSERT_TRUE(empty_loader.initialize({}, 1, 4, false, &error)) << error;
    ValidationResult result;
    EXPECT_FALSE(ValidationEvaluator::evaluate(model, empty_loader, parameters, result, &error));
    EXPECT_FALSE(error.empty());

    TrainingBatchLoader loader;
    ASSERT_TRUE(loader.initialize({0, 1, 2, 0}, 1, 4, true, &error)) << error;
    ASSERT_NE(parameters.find("embedding.weight"), nullptr);
    parameters.find("embedding.weight")->value.data()[0] = std::numeric_limits<float>::quiet_NaN();
    error.clear();
    EXPECT_FALSE(ValidationEvaluator::evaluate(model, loader, parameters, result, &error));
    EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace attention
