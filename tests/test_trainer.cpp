#include "attention/checkpoint.h"
#include "attention/trainer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_training_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

TEST(TrainerTest, ReducesCausalLossAndProducesFiniteParameters) {
    const TransformerConfig config = make_training_config();
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(config, parameters, &error)) << error;
    ASSERT_TRUE(parameters.initialize(17, &error)) << error;
    const std::vector<std::size_t> tokens{0, 1, 2, 0};
    SgdOptimizer optimizer(0.05f);
    TrainingStepResult result;
    ASSERT_TRUE(Trainer::step(model, tokens, 1, 4, parameters, optimizer, result, &error)) << error;
    EXPECT_TRUE(std::isfinite(result.loss_before));
    EXPECT_TRUE(std::isfinite(result.loss_after));
    EXPECT_LT(result.loss_after, result.loss_before);
    EXPECT_TRUE(parameters.all_finite());
}

TEST(TrainerTest, ReloadsUpdatedWeightsWithExactLossEquivalence) {
    const TransformerConfig config = make_training_config();
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(config, parameters, &error)) << error;
    ASSERT_TRUE(parameters.initialize(23, &error)) << error;
    const std::vector<std::size_t> tokens{0, 1, 2, 0};
    SgdOptimizer optimizer(0.02f);
    TrainingStepResult result;
    ASSERT_TRUE(Trainer::step(model, tokens, 1, 4, parameters, optimizer, result, &error)) << error;

    std::string checkpoint;
    ASSERT_TRUE(TransformerCheckpoint::serialize(config, parameters, checkpoint, &error)) << error;
    TransformerModel reloaded;
    ParameterStore reloaded_parameters;
    ASSERT_TRUE(TransformerCheckpoint::load(checkpoint, reloaded, reloaded_parameters, &error)) << error;
    float reloaded_loss = 0.0f;
    ASSERT_TRUE(reloaded.causal_loss(tokens, 1, 4, reloaded_parameters, reloaded_loss, &error)) << error;
    EXPECT_FLOAT_EQ(reloaded_loss, result.loss_after);
}

TEST(AdamWOptimizerTest, PerformsFiniteAdaptiveUpdates) {
    const TransformerConfig config = make_training_config();
    ParameterStore parameters;
    ASSERT_TRUE(TransformerModel().register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(41));
    Parameter* embedding = parameters.find("embedding.weight");
    ASSERT_NE(embedding, nullptr);
    const float value_before = embedding->value.data()[0];
    for (Parameter& parameter : parameters.parameters()) {
        for (std::size_t index = 0; index < parameter.gradient.size(); ++index) {
            parameter.gradient.data()[index] = 0.25f;
        }
    }
    AdamWOptimizer optimizer(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 1.0f);
    std::string error;
    ASSERT_TRUE(optimizer.step(parameters, &error)) << error;
    EXPECT_EQ(optimizer.step_count(), 1u);
    EXPECT_TRUE(parameters.all_finite());
    EXPECT_NE(embedding->value.data()[0], value_before);
}

TEST(SgdOptimizerTest, ClipsGlobalGradientNorm) {
    const TransformerConfig config = make_training_config();
    ParameterStore parameters;
    ASSERT_TRUE(TransformerModel().register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(37));
    for (Parameter& parameter : parameters.parameters()) {
        for (std::size_t index = 0; index < parameter.gradient.size(); ++index) {
            parameter.gradient.data()[index] = 1.0f;
        }
    }
    const float norm_before = parameters.gradient_l2_norm();
    ASSERT_GT(norm_before, 0.5f);
    Parameter* embedding = parameters.find("embedding.weight");
    ASSERT_NE(embedding, nullptr);
    const float value_before = embedding->value.data()[0];
    SgdOptimizer optimizer(1.0f, 0.5f);
    std::string error;
    ASSERT_TRUE(optimizer.step(parameters, &error)) << error;
    const float expected = value_before - 0.5f / norm_before;
    EXPECT_NEAR(embedding->value.data()[0], expected, 1e-6f);
}

TEST(TrainerTest, RejectsInvalidBatchShape) {
    const TransformerConfig config = make_training_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(31));
    TrainingStepResult result;
    std::string error;
    EXPECT_FALSE(Trainer::step(model, {0, 1, 2, 0}, 0, 4, parameters, SgdOptimizer(0.01f), result, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(Trainer::step(model, {0, 1, 2, 0}, 1, 1, parameters, SgdOptimizer(0.01f), result, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(Trainer::step(model, {0, 1, 2}, 1, 4, parameters, SgdOptimizer(0.01f), result, &error));
    EXPECT_FALSE(error.empty());
}

TEST(TrainerTest, RejectsInvalidLearningRateAndGradient) {
    const TransformerConfig config = make_training_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(31));
    TrainingStepResult result;
    std::string error;
    EXPECT_FALSE(Trainer::step(model, {0, 1, 2, 0}, 1, 4, parameters, SgdOptimizer(0.0f), result, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(Trainer::step(model, {0, 1, 2, 0}, 1, 4, parameters, SgdOptimizer(std::numeric_limits<float>::quiet_NaN()), result, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    ASSERT_NE(parameters.find("embedding.weight"), nullptr);
    parameters.find("embedding.weight")->gradient.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(SgdOptimizer(0.01f).step(parameters, &error));
    EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace attention
