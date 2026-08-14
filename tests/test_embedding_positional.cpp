#include <gtest/gtest.h>

#include "attention/parameter_store.h"
#include "attention/positional_encoding.h"
#include "attention/token_embedding.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TransformerConfig make_embedding_config() {
    TransformerConfig config;
    config.vocabulary_size = 16;
    config.context_length = 8;
    config.layer_count = 1;
    config.hidden_size = 4;
    config.attention_head_count = 2;
    config.feed_forward_size = 16;
    return config;
}

TEST(TokenEmbeddingTest, RegistersStableParameterAndCopiesRows) {
    const TransformerConfig config = make_embedding_config();
    ParameterStore parameters;
    TokenEmbedding embedding;
    ASSERT_TRUE(embedding.register_parameters(config, parameters));
    EXPECT_EQ(embedding.parameter_name(), "embedding.weight");
    EXPECT_EQ(embedding.vocabulary_size(), 16u);
    EXPECT_EQ(embedding.hidden_size(), 4u);

    Parameter* weights = parameters.find("embedding.weight");
    ASSERT_NE(weights, nullptr);
    for (std::size_t row = 0; row < config.vocabulary_size; ++row) {
        for (std::size_t column = 0; column < config.hidden_size; ++column) {
            weights->value.data()[row * config.hidden_size + column] =
                static_cast<float>(row * 10 + column);
        }
    }

    Tensor output;
    const std::vector<std::size_t> token_ids = {2, 5, 2, 9};
    std::string error;
    ASSERT_TRUE(embedding.forward(token_ids, 2, 2, parameters, output, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 2, 4}));
    EXPECT_FLOAT_EQ(output.data()[0], 20.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 21.0f);
    EXPECT_FLOAT_EQ(output.data()[4], 50.0f);
    EXPECT_FLOAT_EQ(output.data()[8], 20.0f);
    EXPECT_FLOAT_EQ(output.data()[12], 90.0f);
}

TEST(TokenEmbeddingTest, RejectsWrongShapeAndOutOfRangeTokens) {
    const TransformerConfig config = make_embedding_config();
    ParameterStore parameters;
    TokenEmbedding embedding;
    ASSERT_TRUE(embedding.register_parameters(config, parameters));
    Tensor output;
    std::string error;

    EXPECT_FALSE(embedding.forward({1, 2, 3}, 2, 2, parameters, output, &error));
    EXPECT_EQ(error, "token ID count does not match batch shape");
    EXPECT_FALSE(embedding.forward({1, 16, 3, 4}, 2, 2, parameters, output, &error));
    EXPECT_EQ(error, "token ID is outside the vocabulary");
    EXPECT_FALSE(embedding.forward({1}, 0, 1, parameters, output, &error));
    EXPECT_EQ(error, "batch size and sequence length must be positive");
}

TEST(SinusoidalPositionEncodingTest, ProducesDeterministicExpectedValues) {
    Tensor input;
    ASSERT_TRUE(input.reset({1, 2, 4}));
    input.fill(0.0f);
    SinusoidalPositionEncoding encoding;
    ASSERT_TRUE(encoding.reset(8, 4));

    Tensor first;
    Tensor second;
    ASSERT_TRUE(encoding.apply(input, first));
    ASSERT_TRUE(encoding.apply(input, second));
    EXPECT_EQ(first.shape(), (std::vector<std::size_t>{1, 2, 4}));
    EXPECT_FLOAT_EQ(first.data()[0], 0.0f);
    EXPECT_FLOAT_EQ(first.data()[1], 1.0f);
    EXPECT_NEAR(first.data()[4], std::sin(1.0), 1e-6f);
    EXPECT_NEAR(first.data()[5], std::cos(1.0), 1e-6f);
    for (std::size_t index = 0; index < first.size(); ++index) {
        EXPECT_FLOAT_EQ(first.data()[index], second.data()[index]);
    }
}

TEST(SinusoidalPositionEncodingTest, AbsoluteOffsetMatchesFullSequencePositions) {
    SinusoidalPositionEncoding encoding;
    ASSERT_TRUE(encoding.reset(16, 4));
    Tensor full_input;
    ASSERT_TRUE(full_input.reset({1, 4, 4}));
    full_input.fill(0.0f);
    Tensor full_output;
    ASSERT_TRUE(encoding.apply(full_input, full_output));

    Tensor chunk_input;
    ASSERT_TRUE(chunk_input.reset({1, 2, 4}));
    chunk_input.fill(0.0f);
    Tensor chunk_output;
    ASSERT_TRUE(encoding.apply_at(chunk_input, 2, chunk_output));
    for (std::size_t index = 0; index < chunk_output.size(); ++index) {
        EXPECT_FLOAT_EQ(chunk_output.data()[index], full_output.data()[8 + index]);
    }
}

TEST(SinusoidalPositionEncodingTest, SupportsOddHiddenSizeAndRejectsInvalidInputs) {
    SinusoidalPositionEncoding encoding;
    std::string error;
    EXPECT_FALSE(encoding.reset(0, 4, &error));
    EXPECT_EQ(error, "context length and hidden size must be positive");
    ASSERT_TRUE(encoding.reset(2, 3, &error));

    Tensor input;
    ASSERT_TRUE(input.reset({1, 3, 3}));
    Tensor output;
    EXPECT_FALSE(encoding.apply(input, output, &error));
    EXPECT_EQ(error, "absolute positional range exceeds positional-encoding context length");

    ASSERT_TRUE(input.reset({1, 2, 3}));
    input.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(encoding.apply(input, output, &error));
    EXPECT_EQ(error, "positional encoding input contains NaN or infinity");

    ASSERT_TRUE(input.reset({1, 2, 4}));
    input.fill(0.0f);
    EXPECT_FALSE(encoding.apply(input, output, &error));
    EXPECT_EQ(error, "positional encoding input shape does not match hidden size");
}

} // namespace
} // namespace attention
