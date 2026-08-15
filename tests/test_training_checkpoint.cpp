#include "attention/training_checkpoint.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig resume_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

TEST(TrainingCheckpointTest, PreservesModelAndProgressExactly) {
    const TransformerConfig config = resume_config();
    TransformerModel model;
    ParameterStore parameters;
    std::string error;
    ASSERT_TRUE(model.register_parameters(config, parameters, &error)) << error;
    ASSERT_TRUE(parameters.initialize(53, &error)) << error;
    const std::vector<std::size_t> tokens{0, 1, 2, 0};
    float original_loss = 0.0f;
    ASSERT_TRUE(model.causal_loss(tokens, 1, 4, parameters, original_loss, &error)) << error;

    TrainingProgress progress;
    progress.run_id = "stage0-resume";
    progress.dataset_id = "stage0.debug";
    progress.dataset_revision = "fixed-v1";
    progress.global_step = 7;
    progress.tokens_processed = 28;
    progress.next_batch_index = 7;
    progress.learning_rate = 0.02f;
    std::string checkpoint;
    ASSERT_TRUE(TrainingCheckpoint::serialize(config, parameters, progress, checkpoint, &error)) << error;

    TransformerModel reloaded;
    ParameterStore reloaded_parameters;
    TrainingProgress reloaded_progress;
    ASSERT_TRUE(TrainingCheckpoint::load(checkpoint, reloaded, reloaded_parameters, reloaded_progress, &error)) << error;
    EXPECT_EQ(reloaded_progress.run_id, progress.run_id);
    EXPECT_EQ(reloaded_progress.dataset_id, progress.dataset_id);
    EXPECT_EQ(reloaded_progress.dataset_revision, progress.dataset_revision);
    EXPECT_EQ(reloaded_progress.global_step, progress.global_step);
    EXPECT_EQ(reloaded_progress.tokens_processed, progress.tokens_processed);
    EXPECT_EQ(reloaded_progress.next_batch_index, progress.next_batch_index);
    EXPECT_FLOAT_EQ(reloaded_progress.learning_rate, progress.learning_rate);
    float reloaded_loss = 0.0f;
    ASSERT_TRUE(reloaded.causal_loss(tokens, 1, 4, reloaded_parameters, reloaded_loss, &error)) << error;
    EXPECT_FLOAT_EQ(reloaded_loss, original_loss);
}

TEST(TrainingCheckpointTest, RejectsInvalidProgressAndTrailingData) {
    const TransformerConfig config = resume_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(59));
    TrainingProgress progress;
    progress.run_id = "run";
    progress.dataset_id = "data";
    progress.dataset_revision = "rev";
    progress.learning_rate = 0.01f;
    std::string checkpoint;
    std::string error;
    ASSERT_TRUE(TrainingCheckpoint::serialize(config, parameters, progress, checkpoint, &error)) << error;

    TrainingProgress invalid = progress;
    invalid.learning_rate = 0.0f;
    EXPECT_FALSE(TrainingCheckpoint::serialize(config, parameters, invalid, checkpoint, &error));
    EXPECT_FALSE(error.empty());

    TransformerModel reloaded;
    ParameterStore reloaded_parameters;
    TrainingProgress reloaded_progress;
    error.clear();
    EXPECT_FALSE(TrainingCheckpoint::load(checkpoint + "trailing", reloaded, reloaded_parameters,
                                          reloaded_progress, &error));
    EXPECT_FALSE(error.empty());
}

TEST(TrainingCheckpointTest, RequiresFreshModelAndStore) {
    const TransformerConfig config = resume_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(61));
    TrainingProgress progress{"run", "data", "rev", 0, 0, 0, 0.01f};
    std::string checkpoint;
    ASSERT_TRUE(TrainingCheckpoint::serialize(config, parameters, progress, checkpoint));
    TrainingProgress loaded;
    std::string error;
    EXPECT_FALSE(TrainingCheckpoint::load(checkpoint, model, parameters, loaded, &error));
    EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace attention
