#include <gtest/gtest.h>

#include "attention/token_embedding.h"
#include "attention/vocabulary_projection.h"

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_config(bool tied) {
    TransformerConfig config;
    config.vocabulary_size = 3;
    config.context_length = 8;
    config.layer_count = 1;
    config.hidden_size = 2;
    config.attention_head_count = 1;
    config.feed_forward_size = 4;
    config.tie_embeddings = tied;
    return config;
}

void set_weights(Tensor& weight) {
    weight.data()[0] = 1.0f;
    weight.data()[1] = 2.0f;
    weight.data()[2] = -1.0f;
    weight.data()[3] = 0.5f;
    weight.data()[4] = 0.25f;
    weight.data()[5] = -2.0f;
}

TEST(VocabularyProjectionTest, TiedHeadUsesEmbeddingWeightAndComputesSequenceLogits) {
    ParameterStore parameters;
    TokenEmbedding embedding;
    ASSERT_TRUE(embedding.register_parameters(make_config(true), parameters));
    VocabularyProjection projection;
    ASSERT_TRUE(projection.register_parameters(make_config(true), parameters));
    EXPECT_TRUE(projection.tied_embeddings());
    EXPECT_EQ(parameters.names(), (std::vector<std::string>{
        "embedding.weight", "lm_head.bias"}));
    set_weights(parameters.find("embedding.weight")->value);
    parameters.find("lm_head.bias")->value.fill(0.0f);

    Tensor hidden;
    ASSERT_TRUE(hidden.reset({1, 2, 2}));
    hidden.data()[0] = 2.0f;
    hidden.data()[1] = 3.0f;
    hidden.data()[2] = 1.0f;
    hidden.data()[3] = -1.0f;
    Tensor logits;
    std::string error;
    ASSERT_TRUE(projection.forward(hidden, parameters, logits, &error)) << error;
    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{1, 2, 3}));
    EXPECT_FLOAT_EQ(logits.data()[0], 8.0f);
    EXPECT_FLOAT_EQ(logits.data()[1], -0.5f);
    EXPECT_FLOAT_EQ(logits.data()[2], -5.5f);
    EXPECT_FLOAT_EQ(logits.data()[3], -1.0f);
    EXPECT_FLOAT_EQ(logits.data()[4], -1.5f);
    EXPECT_FLOAT_EQ(logits.data()[5], 2.25f);
}

TEST(VocabularyProjectionTest, UntiedHeadRegistersSeparateWeightAndLastTokenLogits) {
    ParameterStore parameters;
    VocabularyProjection projection;
    ASSERT_TRUE(projection.register_parameters(make_config(false), parameters));
    EXPECT_FALSE(projection.tied_embeddings());
    EXPECT_EQ(parameters.names(), (std::vector<std::string>{
        "lm_head.bias", "lm_head.weight"}));
    set_weights(parameters.find("lm_head.weight")->value);
    parameters.find("lm_head.bias")->value.fill(1.0f);

    Tensor hidden;
    ASSERT_TRUE(hidden.reset({2, 1, 2}));
    hidden.data()[0] = 2.0f;
    hidden.data()[1] = 3.0f;
    hidden.data()[2] = 1.0f;
    hidden.data()[3] = -1.0f;
    Tensor logits;
    ASSERT_TRUE(projection.forward_last(hidden, parameters, logits));
    EXPECT_EQ(logits.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_FLOAT_EQ(logits.data()[0], 9.0f);
    EXPECT_FLOAT_EQ(logits.data()[1], 0.5f);
    EXPECT_FLOAT_EQ(logits.data()[2], -4.5f);
    EXPECT_FLOAT_EQ(logits.data()[3], 0.0f);
    EXPECT_FLOAT_EQ(logits.data()[4], -0.5f);
    EXPECT_FLOAT_EQ(logits.data()[5], 3.25f);
}

TEST(VocabularyProjectionTest, RejectsMissingTiedEmbeddingDuplicateAndNonfiniteInputs) {
    ParameterStore parameters;
    VocabularyProjection tied;
    std::string error;
    EXPECT_FALSE(tied.register_parameters(make_config(true), parameters, &error));
    EXPECT_EQ(error, "tied vocabulary projection requires embedding.weight");

    ParameterStore untied_parameters;
    VocabularyProjection first;
    VocabularyProjection second;
    ASSERT_TRUE(first.register_parameters(make_config(false), untied_parameters, &error)) << error;
    EXPECT_FALSE(second.register_parameters(make_config(false), untied_parameters, &error));
    EXPECT_EQ(error, "vocabulary projection parameter name already exists");

    Tensor hidden;
    ASSERT_TRUE(hidden.reset({1, 1, 2}));
    hidden.data()[0] = std::numeric_limits<float>::quiet_NaN();
    Tensor logits;
    EXPECT_FALSE(first.forward_last(hidden, untied_parameters, logits, &error));
    EXPECT_EQ(error, "last-token projection requires a finite rank-3 tensor with sequence length one");
}

TEST(AutoregressiveLogitsTest, ExposesConfiguredLastTokenPath) {
    ParameterStore parameters;
    AutoregressiveLogits logits_head;
    ASSERT_TRUE(logits_head.register_parameters(make_config(false), parameters));
    EXPECT_EQ(logits_head.vocabulary_size(), 3u);
    EXPECT_EQ(logits_head.hidden_size(), 2u);
    EXPECT_FALSE(logits_head.tied_embeddings());
}

} // namespace
} // namespace attention
