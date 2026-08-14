#include "attention/checkpoint.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_checkpoint_config() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    return config;
}

TEST(CheckpointTest, DeterministicSerializationAndReloadPreserveLogits) {
    const TransformerConfig config = make_checkpoint_config();
    TransformerModel original;
    ParameterStore original_parameters;
    ASSERT_TRUE(original.register_parameters(config, original_parameters));
    ASSERT_TRUE(original_parameters.initialize(1234));

    std::string first;
    std::string second;
    std::string error;
    ASSERT_TRUE(TransformerCheckpoint::serialize(config, original_parameters, first, &error)) << error;
    ASSERT_TRUE(TransformerCheckpoint::serialize(config, original_parameters, second, &error)) << error;
    EXPECT_EQ(first, second);
    EXPECT_NE(first.find("attention.checkpoint.v1"), std::string::npos);
    EXPECT_NE(first.find("parameter_count"), std::string::npos);

    const std::vector<std::size_t> tokens{0, 1, 2};
    Tensor original_logits;
    ASSERT_TRUE(original.forward(tokens, 1, 3, original_parameters, original_logits, &error)) << error;

    TransformerModel reloaded;
    ParameterStore reloaded_parameters;
    ASSERT_TRUE(TransformerCheckpoint::load(first, reloaded, reloaded_parameters, &error)) << error;
    Tensor reloaded_logits;
    ASSERT_TRUE(reloaded.forward(tokens, 1, 3, reloaded_parameters, reloaded_logits, &error)) << error;
    ASSERT_EQ(original_logits.shape(), reloaded_logits.shape());
    for (std::size_t index = 0; index < original_logits.size(); ++index) {
        EXPECT_FLOAT_EQ(original_logits.data()[index], reloaded_logits.data()[index]);
    }
}

TEST(CheckpointTest, RejectsMalformedConfigurationAndParameterPayloads) {
    const TransformerConfig config = make_checkpoint_config();
    TransformerModel model;
    ParameterStore parameters;
    ASSERT_TRUE(model.register_parameters(config, parameters));
    ASSERT_TRUE(parameters.initialize(9));
    std::string checkpoint;
    ASSERT_TRUE(TransformerCheckpoint::serialize(config, parameters, checkpoint));
    std::string error;

    TransformerModel bad_magic_model;
    ParameterStore bad_magic_parameters;
    std::string bad_magic = checkpoint;
    bad_magic.replace(0, std::string("attention.checkpoint.v1").size(), "wrong.checkpoint.v1");
    EXPECT_FALSE(TransformerCheckpoint::load(bad_magic, bad_magic_model, bad_magic_parameters, &error));

    TransformerModel bad_config_model;
    ParameterStore bad_config_parameters;
    std::string bad_config = checkpoint;
    const std::size_t config_start = bad_config.find("attention.transformer_config.v1");
    ASSERT_NE(config_start, std::string::npos);
    bad_config.replace(config_start, std::string("attention.transformer_config.v1").size(), "broken.config.v1");
    EXPECT_FALSE(TransformerCheckpoint::load(bad_config, bad_config_model, bad_config_parameters, &error));

    TransformerModel trailing_model;
    ParameterStore trailing_parameters;
    EXPECT_FALSE(TransformerCheckpoint::load(checkpoint + "trailing\n", trailing_model, trailing_parameters, &error));

    TransformerModel duplicate_model;
    ParameterStore duplicate_parameters;
    std::string duplicate = checkpoint;
    const std::size_t first_name = duplicate.find("name_bytes\n");
    ASSERT_NE(first_name, std::string::npos);
    const std::size_t second_name = duplicate.find("name_bytes\n", first_name + 1);
    ASSERT_NE(second_name, std::string::npos);
    const std::size_t second_name_length = duplicate.find('\n', second_name + 11);
    ASSERT_NE(second_name_length, std::string::npos);
    const std::size_t second_name_payload = second_name_length + 1;
    const std::size_t first_name_length = duplicate.find('\n', first_name + 11);
    ASSERT_NE(first_name_length, std::string::npos);
    const std::size_t first_name_payload = first_name_length + 1;
    const std::size_t first_name_end = duplicate.find('\n', first_name_payload);
    const std::string first_name_text = duplicate.substr(first_name_payload, first_name_end - first_name_payload);
    const std::size_t second_name_end = duplicate.find('\n', second_name_payload);
    duplicate.replace(second_name_payload, second_name_end - second_name_payload,
                      first_name_text);
    EXPECT_FALSE(TransformerCheckpoint::load(duplicate, duplicate_model, duplicate_parameters, &error));
}

TEST(CheckpointTest, RejectsLoadingIntoNonfreshModelOrParameterStore) {
    const TransformerConfig config = make_checkpoint_config();
    TransformerModel source;
    ParameterStore source_parameters;
    ASSERT_TRUE(source.register_parameters(config, source_parameters));
    std::string checkpoint;
    ASSERT_TRUE(TransformerCheckpoint::serialize(config, source_parameters, checkpoint));

    TransformerModel existing_model;
    ParameterStore existing_parameters;
    ASSERT_TRUE(existing_model.register_parameters(config, existing_parameters));
    std::string error;
    EXPECT_FALSE(TransformerCheckpoint::load(checkpoint, existing_model, existing_parameters, &error));

    TransformerModel fresh_model;
    ParameterStore nonempty_parameters;
    ASSERT_TRUE(nonempty_parameters.add("unexpected", {1}));
    EXPECT_FALSE(TransformerCheckpoint::load(checkpoint, fresh_model, nonempty_parameters, &error));
}

} // namespace
} // namespace attention
