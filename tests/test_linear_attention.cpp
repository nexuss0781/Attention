#include <gtest/gtest.h>

#include "attention/linear_attention.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

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
