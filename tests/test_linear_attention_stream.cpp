#include <gtest/gtest.h>

#include "attention/linear_attention.h"

#include <string>
#include <vector>

namespace attention {
namespace {

Tensor make_tensor(const std::vector<float>& values, std::size_t sequence_length) {
    Tensor tensor;
    EXPECT_TRUE(tensor.reset({1, sequence_length, 2}));
    for (std::size_t index = 0; index < values.size(); ++index) {
        tensor.data()[index] = values[index];
    }
    return tensor;
}

Tensor slice_tensor(const Tensor& source, std::size_t start, std::size_t length) {
    Tensor result;
    EXPECT_TRUE(result.reset({1, length, 2}));
    for (std::size_t index = 0; index < length * 2; ++index) {
        result.data()[index] = source.data()[start * 2 + index];
    }
    return result;
}

TEST(LinearAttentionStreamTest, ChunkedAppendMatchesSinglePass) {
    const Tensor query = make_tensor({0.0f, 0.2f, 0.4f, -0.1f, 0.8f, 0.3f, -0.2f, 0.6f}, 4);
    const Tensor key = make_tensor({0.1f, 0.0f, -0.3f, 0.2f, 0.5f, -0.4f, 0.7f, 0.1f}, 4);
    const Tensor value = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 4);

    LinearCausalAttention one_pass_attention;
    ASSERT_TRUE(one_pass_attention.reset(6, 2));
    Tensor one_pass;
    ASSERT_TRUE(one_pass_attention.forward(query, key, value, one_pass));

    LinearAttentionState stream;
    ASSERT_TRUE(one_pass_attention.create_stream(1, stream));
    EXPECT_EQ(stream.tokens_processed(), 0u);
    EXPECT_EQ(stream.state_bytes(), 64u);
    Tensor first_query = slice_tensor(query, 0, 2);
    Tensor first_key = slice_tensor(key, 0, 2);
    Tensor first_value = slice_tensor(value, 0, 2);
    Tensor first_output;
    ASSERT_TRUE(stream.append(first_query, first_key, first_value, first_output));
    EXPECT_EQ(stream.tokens_processed(), 2u);

    Tensor second_query = slice_tensor(query, 2, 2);
    Tensor second_key = slice_tensor(key, 2, 2);
    Tensor second_value = slice_tensor(value, 2, 2);
    Tensor second_output;
    ASSERT_TRUE(stream.append(second_query, second_key, second_value, second_output));
    EXPECT_EQ(stream.tokens_processed(), 4u);

    for (std::size_t index = 0; index < first_output.size(); ++index) {
        EXPECT_NEAR(first_output.data()[index], one_pass.data()[index], 1e-6f);
    }
    for (std::size_t index = 0; index < second_output.size(); ++index) {
        EXPECT_NEAR(second_output.data()[index], one_pass.data()[4 + index], 1e-6f);
    }
}

TEST(LinearAttentionStreamTest, BillionTokenLogicalContextUsesConstantState) {
    LinearAttentionState stream;
    ASSERT_TRUE(stream.reset(1000000000ULL, 1, 2));
    EXPECT_EQ(stream.context_length(), 1000000000ULL);
    EXPECT_EQ(stream.tokens_processed(), 0u);
    EXPECT_EQ(stream.state_bytes(), 64u);

    Tensor query = make_tensor({0.0f, 0.0f}, 1);
    Tensor key = make_tensor({0.0f, 0.0f}, 1);
    Tensor value = make_tensor({3.0f, 4.0f}, 1);
    Tensor output;
    std::string error;
    ASSERT_TRUE(stream.append(query, key, value, output, &error)) << error;
    EXPECT_EQ(stream.tokens_processed(), 1u);
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{1, 1, 2}));
    EXPECT_FLOAT_EQ(output.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 4.0f);
}

TEST(LinearAttentionStreamTest, RejectsContextOverflowAndShapeMismatch) {
    LinearAttentionState stream;
    ASSERT_TRUE(stream.reset(3, 1, 2));
    Tensor query = make_tensor({0.0f, 0.0f, 0.0f, 0.0f}, 2);
    Tensor key = make_tensor({0.0f, 0.0f, 0.0f, 0.0f}, 2);
    Tensor value = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2);
    Tensor output;
    ASSERT_TRUE(stream.append(query, key, value, output));
    EXPECT_EQ(stream.tokens_processed(), 2u);
    EXPECT_FALSE(stream.append(query, key, value, output));
    EXPECT_EQ(stream.tokens_processed(), 2u);
}

} // namespace
} // namespace attention
