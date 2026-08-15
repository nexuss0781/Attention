#include <gtest/gtest.h>

#include "attention/linear_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace attention {
namespace {

void scalar_reference(const Tensor& query,
                      const Tensor& key,
                      const Tensor& value,
                      std::size_t head_count,
                      float epsilon,
                      Tensor& output) {
    const auto& shape = query.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    const std::size_t hidden_size = shape[2];
    const std::size_t head_size = hidden_size / head_count;
    ASSERT_TRUE(output.reset(shape));
    std::vector<double> state(head_count * head_size * head_size, 0.0);
    std::vector<double> normalizer(hidden_size, 0.0);
    std::vector<float> query_features(hidden_size);
    std::vector<float> key_features(hidden_size);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        std::fill(state.begin(), state.end(), 0.0);
        std::fill(normalizer.begin(), normalizer.end(), 0.0);
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t row_offset = (batch * sequence_length + position) * hidden_size;
            for (std::size_t channel = 0; channel < hidden_size; ++channel) {
                key_features[channel] = std::exp(std::clamp(key.data()[row_offset + channel], -20.0f, 20.0f));
                normalizer[channel] += static_cast<double>(key_features[channel]);
                query_features[channel] = std::exp(std::clamp(query.data()[row_offset + channel], -20.0f, 20.0f));
            }
            for (std::size_t head = 0; head < head_count; ++head) {
                const std::size_t offset = head * head_size;
                double* head_state = state.data() + head * head_size * head_size;
                for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                    double* state_row = head_state + key_channel * head_size;
                    for (std::size_t value_channel = 0; value_channel < head_size; ++value_channel) {
                        state_row[value_channel] += static_cast<double>(key_features[offset + key_channel]) *
                            static_cast<double>(value.data()[row_offset + offset + value_channel]);
                    }
                }
                double denominator = 0.0;
                for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                    denominator += static_cast<double>(query_features[offset + key_channel]) *
                        normalizer[offset + key_channel];
                }
                const double safe_denominator = std::max(denominator, static_cast<double>(epsilon));
                for (std::size_t value_channel = 0; value_channel < head_size; ++value_channel) {
                    double numerator = 0.0;
                    for (std::size_t key_channel = 0; key_channel < head_size; ++key_channel) {
                        numerator += static_cast<double>(query_features[offset + key_channel]) *
                            head_state[key_channel * head_size + value_channel];
                    }
                    output.data()[row_offset + offset + value_channel] =
                        static_cast<float>(numerator / safe_denominator);
                }
            }
        }
    }
}

TEST(LinearCausalAttentionTest, CausalUniformFeatureMapProducesStreamingAverage) {
    LinearCausalAttention attention;
    ASSERT_TRUE(attention.reset(8, 2));

    Tensor query;
    Tensor key;
    Tensor value;
    ASSERT_TRUE(query.reset({1, 3, 2}));
    ASSERT_TRUE(key.reset({1, 3, 2}));
    ASSERT_TRUE(value.reset({1, 3, 2}));
    query.fill(0.0f);
    key.fill(0.0f);
    value.data()[0] = 1.0f;
    value.data()[1] = 2.0f;
    value.data()[2] = 3.0f;
    value.data()[3] = 4.0f;
    value.data()[4] = 5.0f;
    value.data()[5] = 6.0f;

    Tensor output;
    std::string error;
    ASSERT_TRUE(attention.forward(query, key, value, output, &error)) << error;
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{1, 3, 2}));
    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[4], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[5], 4.0f);
}

TEST(LinearCausalAttentionTest, IsDeterministicAndUsesDimensionBoundedState) {
    LinearCausalAttention attention;
    ASSERT_TRUE(attention.reset(128, 4));
    EXPECT_EQ(attention.state_bytes(2), 320u);

    Tensor query;
    Tensor key;
    Tensor value;
    ASSERT_TRUE(query.reset({2, 5, 4}));
    ASSERT_TRUE(key.reset({2, 5, 4}));
    ASSERT_TRUE(value.reset({2, 5, 4}));
    for (std::size_t index = 0; index < query.size(); ++index) {
        query.data()[index] = static_cast<float>(index % 7) * 0.1f;
        key.data()[index] = static_cast<float>(index % 5) * -0.1f;
        value.data()[index] = static_cast<float>(index % 11) * 0.2f;
    }

    Tensor first;
    Tensor second;
    ASSERT_TRUE(attention.forward(query, key, value, first));
    ASSERT_TRUE(attention.forward(query, key, value, second));
    ASSERT_TRUE(first.all_finite());
    for (std::size_t index = 0; index < first.size(); ++index) {
        EXPECT_FLOAT_EQ(first.data()[index], second.data()[index]);
    }
}

TEST(LinearCausalAttentionTest, EigenKernelMatchesScalarReference) {
    constexpr std::size_t batch_size = 2;
    constexpr std::size_t sequence_length = 17;
    constexpr std::size_t hidden_size = 8;
    constexpr std::size_t head_count = 2;
    Tensor query;
    Tensor key;
    Tensor value;
    ASSERT_TRUE(query.reset({batch_size, sequence_length, hidden_size}));
    ASSERT_TRUE(key.reset({batch_size, sequence_length, hidden_size}));
    ASSERT_TRUE(value.reset({batch_size, sequence_length, hidden_size}));
    std::mt19937 generator(20260815);
    std::uniform_real_distribution<float> distribution(-0.75f, 0.75f);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query.data()[index] = distribution(generator);
        key.data()[index] = distribution(generator);
        value.data()[index] = distribution(generator);
    }
    LinearCausalAttention attention;
    ASSERT_TRUE(attention.reset(sequence_length, hidden_size, 1e-6f, nullptr, head_count));
    Tensor actual;
    ASSERT_TRUE(attention.forward(query, key, value, actual));
    Tensor expected;
    scalar_reference(query, key, value, head_count, 1e-6f, expected);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual.data()[index], expected.data()[index], 1e-5f) << index;
    }
}

TEST(LinearCausalAttentionTest, HeadPartitionReducesStateFootprint) {
    LinearCausalAttention single_head;
    LinearCausalAttention two_heads;
    std::string error;
    ASSERT_TRUE(single_head.reset(8, 4, 1e-6f, &error)) << error;
    ASSERT_TRUE(two_heads.reset(8, 4, 1e-6f, &error, 2)) << error;
    EXPECT_EQ(single_head.head_count(), 1u);
    EXPECT_EQ(two_heads.head_count(), 2u);
    EXPECT_EQ(single_head.state_bytes(2), 320u);
    EXPECT_EQ(two_heads.state_bytes(2), 192u);
}

TEST(LinearCausalAttentionTest, RejectsInvalidConfigurationShapesAndValues) {
    LinearCausalAttention attention;
    std::string error;
    EXPECT_FALSE(attention.reset(0, 4, 1e-6f, &error));
    EXPECT_EQ(error, "context length and hidden size must be positive");
    EXPECT_FALSE(attention.reset(8, 4, 0.0f, &error));
    EXPECT_EQ(error, "epsilon must be finite and positive");
    ASSERT_TRUE(attention.reset(2, 2, 1e-6f, &error));

    Tensor query;
    Tensor key;
    Tensor value;
    ASSERT_TRUE(query.reset({1, 3, 2}));
    ASSERT_TRUE(key.reset({1, 3, 2}));
    ASSERT_TRUE(value.reset({1, 3, 2}));
    Tensor output;
    EXPECT_FALSE(attention.forward(query, key, value, output, &error));
    EXPECT_EQ(error, "linear attention input shape exceeds context or hidden size");

    ASSERT_TRUE(query.reset({1, 2, 2}));
    ASSERT_TRUE(key.reset({1, 2, 2}));
    ASSERT_TRUE(value.reset({1, 2, 2}));
    query.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(attention.forward(query, key, value, output, &error));
    EXPECT_EQ(error, "linear attention inputs contain NaN or infinity");

    ASSERT_TRUE(query.reset({1, 2, 2}));
    ASSERT_TRUE(key.reset({1, 2, 2}));
    ASSERT_TRUE(value.reset({1, 2, 3}));
    EXPECT_FALSE(attention.forward(query, key, value, output, &error));
    EXPECT_EQ(error, "linear attention input shapes must match");
}

} // namespace
} // namespace attention
