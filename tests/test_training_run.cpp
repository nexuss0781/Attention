#include "attention/training_run.h"
#include "attention/transformer_config.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

namespace attention {
namespace {

TrainingRunMetadata make_metadata() {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 4;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    std::string architecture;
    EXPECT_TRUE(config.serialize(architecture));

    TrainingRunMetadata metadata;
    metadata.run_id = "stage0-debug-001";
    metadata.stage = "stage0_debug";
    metadata.dataset_id = "local.measurement.debug";
    metadata.dataset_revision = "fixed-v1";
    metadata.source_checksums = "sha256:abc123";
    metadata.tokenizer_version = "attention.byte_utf8.v1";
    metadata.tokenizer_vocabulary_size = 260;
    metadata.architecture_serialization = architecture;
    metadata.code_commit = "e9bc6b1";
    metadata.seed = 17;
    metadata.batch_size = 1;
    metadata.sequence_length = 4;
    metadata.learning_rate = 0.05f;
    return metadata;
}

TEST(TrainingRunLoggerTest, SerializesDeterministicallyWithStepRecords) {
    TrainingRunLogger logger;
    std::string error;
    ASSERT_TRUE(logger.initialize(make_metadata(), &error)) << error;
    ASSERT_TRUE(logger.append({0, 0, 0, 4, 1.2f, 1.1f, 0.05f, 0.4f}, &error)) << error;
    ASSERT_TRUE(logger.append({1, 1, 4, 8, 1.1f, 1.0f, 0.05f, 0.3f}, &error)) << error;

    std::string first;
    std::string second;
    ASSERT_TRUE(logger.serialize(first, &error)) << error;
    ASSERT_TRUE(logger.serialize(second, &error)) << error;
    EXPECT_EQ(first, second);
    EXPECT_NE(first.find("attention.training_run.v1"), std::string::npos);
    EXPECT_NE(first.find("gradient_l2_norm"), std::string::npos);
    EXPECT_NE(first.find("architecture_serialization"), std::string::npos);
}

TEST(TrainingRunLoggerTest, RejectsInvalidMetadataAndRecords) {
    TrainingRunLogger logger;
    std::string error;
    TrainingRunMetadata invalid = make_metadata();
    invalid.learning_rate = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(logger.initialize(invalid, &error));
    EXPECT_FALSE(error.empty());

    ASSERT_TRUE(logger.initialize(make_metadata(), &error)) << error;
    EXPECT_FALSE(logger.append({0, 0, 0, 4, std::numeric_limits<float>::quiet_NaN(), 1.1f, 0.05f, 0.4f}, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    ASSERT_TRUE(logger.append({1, 0, 0, 4, 1.2f, 1.1f, 0.05f, 0.4f}, &error)) << error;
    EXPECT_FALSE(logger.append({1, 1, 4, 8, 1.1f, 1.0f, 0.05f, 0.3f}, &error));
    EXPECT_FALSE(error.empty());
}

TEST(TrainingRunLoggerTest, RejectsSerializationBeforeInitialization) {
    TrainingRunLogger logger;
    std::string output;
    std::string error;
    EXPECT_FALSE(logger.serialize(output, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(std::isfinite(1.0f));
}

} // namespace
} // namespace attention
