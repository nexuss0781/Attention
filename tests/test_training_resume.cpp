#include "attention/trainer.h"
#include "attention/training_checkpoint.h"
#include "attention/training_data.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig resume_training_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

bool train_batches(TransformerModel& model,
                   ParameterStore& parameters,
                   TrainingBatchLoader& loader,
                   std::size_t count,
                   std::string* error) {
    SgdOptimizer optimizer(0.05f);
    for (std::size_t step = 0; step < count; ++step) {
        TrainingBatch batch;
        TrainingStepResult result;
        if (!loader.next(batch, error) ||
            !Trainer::step(model, batch.token_ids, batch.batch_size, batch.sequence_length,
                           parameters, optimizer, result, error)) return false;
    }
    return true;
}

TEST(TrainingResumeTest, InterruptedAndUninterruptedRunsMatchExactly) {
    const TransformerConfig config = resume_training_config();
    const std::vector<std::size_t> tokens{
        0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
    std::string error;

    TransformerModel uninterrupted_model;
    ParameterStore uninterrupted_parameters;
    ASSERT_TRUE(uninterrupted_model.register_parameters(config, uninterrupted_parameters, &error)) << error;
    ASSERT_TRUE(uninterrupted_parameters.initialize(71, &error)) << error;
    TrainingBatchLoader uninterrupted_loader;
    ASSERT_TRUE(uninterrupted_loader.initialize(tokens, 1, 4, true, &error)) << error;
    ASSERT_TRUE(train_batches(uninterrupted_model, uninterrupted_parameters,
                              uninterrupted_loader, 4, &error)) << error;

    TransformerModel interrupted_model;
    ParameterStore interrupted_parameters;
    ASSERT_TRUE(interrupted_model.register_parameters(config, interrupted_parameters, &error)) << error;
    ASSERT_TRUE(interrupted_parameters.initialize(71, &error)) << error;
    TrainingBatchLoader interrupted_loader;
    ASSERT_TRUE(interrupted_loader.initialize(tokens, 1, 4, true, &error)) << error;
    ASSERT_TRUE(train_batches(interrupted_model, interrupted_parameters,
                              interrupted_loader, 2, &error)) << error;

    TrainingProgress progress{"stage0-resume", "stage0.debug", "fixed-v1",
                              2, interrupted_loader.tokens_processed(),
                              interrupted_loader.batches_emitted(), 0.05f};
    std::string checkpoint;
    ASSERT_TRUE(TrainingCheckpoint::serialize(config, interrupted_parameters, progress,
                                              checkpoint, &error)) << error;

    TransformerModel resumed_model;
    ParameterStore resumed_parameters;
    TrainingProgress resumed_progress;
    ASSERT_TRUE(TrainingCheckpoint::load(checkpoint, resumed_model, resumed_parameters,
                                         resumed_progress, &error)) << error;
    EXPECT_EQ(resumed_progress.global_step, 2U);
    EXPECT_EQ(resumed_progress.next_batch_index, 2U);

    TrainingBatchLoader resumed_loader;
    ASSERT_TRUE(resumed_loader.initialize(tokens, 1, 4, true, &error)) << error;
    for (std::size_t skipped = 0; skipped < resumed_progress.next_batch_index; ++skipped) {
        TrainingBatch ignored;
        ASSERT_TRUE(resumed_loader.next(ignored, &error)) << error;
    }
    ASSERT_TRUE(train_batches(resumed_model, resumed_parameters,
                              resumed_loader, 2, &error)) << error;

    const std::vector<std::string> names = uninterrupted_parameters.names();
    ASSERT_EQ(names, resumed_parameters.names());
    for (const std::string& name : names) {
        const Parameter* expected = uninterrupted_parameters.find(name);
        const Parameter* actual = resumed_parameters.find(name);
        ASSERT_NE(expected, nullptr);
        ASSERT_NE(actual, nullptr);
        ASSERT_EQ(expected->value.size(), actual->value.size());
        for (std::size_t index = 0; index < expected->value.size(); ++index) {
            EXPECT_FLOAT_EQ(expected->value.data()[index], actual->value.data()[index]) << name << '[' << index << ']';
        }
    }
    float uninterrupted_loss = 0.0f;
    float resumed_loss = 0.0f;
    ASSERT_TRUE(uninterrupted_model.causal_loss({0, 1, 2, 0}, 1, 4,
                                                uninterrupted_parameters, uninterrupted_loss, &error)) << error;
    ASSERT_TRUE(resumed_model.causal_loss({0, 1, 2, 0}, 1, 4,
                                          resumed_parameters, resumed_loss, &error)) << error;
    EXPECT_FLOAT_EQ(uninterrupted_loss, resumed_loss);
}

} // namespace
} // namespace attention
