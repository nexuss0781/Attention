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
    EXPECT_EQ(config.parameter_count(), 156776u);
    EXPECT_EQ(config.parameter_bytes(), 156776u * sizeof(float));
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

namespace attention {
namespace {

TEST(TransformerConfigTest, EstimatesFullInferenceMemoryWithResidentSequenceScaling) {
    const TransformerConfig config = make_valid_config();
    EXPECT_EQ(config.estimated_inference_memory_bytes(2), 1620192u);
    EXPECT_EQ(config.estimated_inference_memory_bytes(2, 16), 816480u);
    EXPECT_EQ(config.estimated_inference_memory_bytes(0), 0u);
    EXPECT_EQ(config.estimated_inference_memory_bytes(2, config.context_length + 1), 0u);
}

TEST(TransformerConfigTest, SerializesDeterministicallyAndRoundTripsAllFields) {
    TransformerConfig config = make_valid_config();
    config.activation = Activation::SiLU;
    config.tie_embeddings = false;
    config.dropout_probability = 0.125f;

    std::string first;
    std::string second;
    std::string error;
    ASSERT_TRUE(config.serialize(first, &error)) << error;
    ASSERT_TRUE(config.serialize(second, &error)) << error;
    EXPECT_EQ(first, second);
    EXPECT_EQ(first,
              "attention.transformer_config.v1\n"
              "vocabulary_size=1000\n"
              "context_length=128\n"
              "layer_count=2\n"
              "hidden_size=64\n"
              "attention_head_count=4\n"
              "feed_forward_size=256\n"
              "activation=1\n"
              "precision=0\n"
              "tie_embeddings=0\n"
              "causal=1\n"
              "dropout_probability=0.125\n");

    TransformerConfig restored;
    ASSERT_TRUE(TransformerConfig::deserialize(first, restored, &error)) << error;
    EXPECT_EQ(restored.vocabulary_size, config.vocabulary_size);
    EXPECT_EQ(restored.context_length, config.context_length);
    EXPECT_EQ(restored.layer_count, config.layer_count);
    EXPECT_EQ(restored.hidden_size, config.hidden_size);
    EXPECT_EQ(restored.attention_head_count, config.attention_head_count);
    EXPECT_EQ(restored.feed_forward_size, config.feed_forward_size);
    EXPECT_EQ(restored.activation, config.activation);
    EXPECT_EQ(restored.precision, config.precision);
    EXPECT_EQ(restored.tie_embeddings, config.tie_embeddings);
    EXPECT_EQ(restored.causal, config.causal);
    EXPECT_FLOAT_EQ(restored.dropout_probability, config.dropout_probability);
}

TEST(TransformerConfigTest, RejectsMalformedOrNonCanonicalSerialization) {
    TransformerConfig restored;
    std::string error;
    EXPECT_FALSE(TransformerConfig::deserialize("", restored, &error));
    EXPECT_FALSE(TransformerConfig::deserialize(
        "attention.transformer_config.v1\n"
        "vocabulary_size=1000\n"
        "context_length=128\n"
        "layer_count=2\n"
        "hidden_size=64\n"
        "attention_head_count=4\n"
        "feed_forward_size=256\n"
        "activation=0\n"
        "precision=0\n"
        "tie_embeddings=1\n"
        "causal=1\n"
        "dropout_probability=0\n"
        "extra=1\n", restored, &error));
    EXPECT_FALSE(TransformerConfig::deserialize(
        "attention.transformer_config.v1\n"
        "vocabulary_size=1000\n"
        "context_length=128\n"
        "layer_count=2\n"
        "hidden_size=64\n"
        "attention_head_count=4\n"
        "feed_forward_size=256\n"
        "activation=9\n"
        "precision=0\n"
        "tie_embeddings=1\n"
        "causal=1\n"
        "dropout_probability=0\n", restored, &error));
}

TEST(TransformerConfigTest, SerializationRejectsInvalidConfig) {
    TransformerConfig invalid = make_valid_config();
    invalid.dropout_probability = std::numeric_limits<float>::quiet_NaN();
    std::string serialized;
    std::string error;
    EXPECT_FALSE(invalid.serialize(serialized, &error));
    EXPECT_TRUE(serialized.empty());
}

TEST(TransformerConfigTest, InferenceMemoryOverflowReturnsZero) {
    TransformerConfig config = make_valid_config();
    config.vocabulary_size = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ(config.estimated_inference_memory_bytes(1), 0u);
}

} // namespace
} // namespace attention
