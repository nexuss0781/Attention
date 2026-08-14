#include <gtest/gtest.h>

#include "attention/transformer_config.h"

#include <limits>
#include <string>

namespace attention {
namespace {

TransformerConfig make_valid_config() {
    TransformerConfig config;
    config.vocabulary_size = 1000;
    config.context_length = 128;
    config.layer_count = 2;
    config.hidden_size = 64;
    config.attention_head_count = 4;
    config.feed_forward_size = 256;
    return config;
}

TEST(TransformerConfigTest, ValidatesAndComputesDerivedValues) {
    const TransformerConfig config = make_valid_config();
    std::string error;
    ASSERT_TRUE(config.validate(&error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(config.head_size(), 16u);
    EXPECT_EQ(config.parameter_count(), 173416u);
    EXPECT_EQ(config.parameter_bytes(), 173416u * sizeof(float));
    EXPECT_EQ(config.estimated_activation_bytes(2), 1572864u);
}

TEST(TransformerConfigTest, RejectsInvalidShape) {
    TransformerConfig config = make_valid_config();
    config.hidden_size = 65;
    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(error, "hidden_size must be divisible by attention_head_count");
}

TEST(TransformerConfigTest, RejectsNonCausalAndUnsupportedPrecision) {
    TransformerConfig config = make_valid_config();
    std::string error;
    config.causal = false;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(error, "the first Attention transformer foundation requires causal attention");

    config.causal = true;
    config.precision = static_cast<ModelPrecision>(99);
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(error, "only F32 is implemented in the first transformer foundation");
}

TEST(TransformerConfigTest, RejectsInvalidDropout) {
    TransformerConfig config = make_valid_config();
    std::string error;
    config.dropout_probability = 1.0f;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(error, "dropout_probability must be finite and in [0, 1)");

    config.dropout_probability = -0.1f;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(error, "dropout_probability must be finite and in [0, 1)");
}

TEST(TransformerConfigTest, TiedAndUntiedOutputHeadsDifferPredictably) {
    TransformerConfig tied = make_valid_config();
    TransformerConfig untied = tied;
    untied.tie_embeddings = false;
    EXPECT_EQ(untied.parameter_count() - tied.parameter_count(),
              tied.vocabulary_size * tied.hidden_size);
}

TEST(TransformerConfigTest, OverflowReturnsZero) {
    TransformerConfig config = make_valid_config();
    config.vocabulary_size = std::numeric_limits<std::size_t>::max();
    std::string error;
    EXPECT_FALSE(config.validate(&error));
    EXPECT_EQ(config.parameter_count(), 0u);
    EXPECT_EQ(config.parameter_bytes(), 0u);
}

} // namespace
} // namespace attention
