#include "attention/hierarchical_summary.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

Tensor make_values(const std::vector<float>& values, std::size_t batch, std::size_t length, std::size_t hidden) {
    Tensor tensor;
    EXPECT_TRUE(tensor.reset({batch, length, hidden}));
    if (values.size() != tensor.size()) return tensor;
    for (std::size_t index = 0; index < values.size(); ++index) tensor.data()[index] = values[index];
    return tensor;
}

TEST(HierarchicalSummaryTest, ProducesDeterministicMultiScaleMeans) {
    HierarchicalSummaryState state;
    std::string error;
    ASSERT_TRUE(state.reset(16, 3, 2, 1, 1, &error)) << error;
    const Tensor input = make_values({1.0f, 3.0f, 5.0f}, 1, 3, 1);
    ASSERT_TRUE(state.append(input, &error)) << error;

    Tensor snapshot;
    ASSERT_TRUE(state.snapshot(snapshot, &error)) << error;
    ASSERT_EQ(snapshot.shape(), (std::vector<std::size_t>{1, 3, 1}));
    EXPECT_FLOAT_EQ(snapshot.data()[0], 5.0f);
    EXPECT_FLOAT_EQ(snapshot.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(snapshot.data()[2], 0.0f);
    EXPECT_EQ(state.tokens_processed(), 3u);
}

TEST(HierarchicalSummaryTest, ChunkedAppendMatchesSingleAppend) {
    const Tensor first = make_values({1.0f, 2.0f, 3.0f}, 1, 3, 1);
    const Tensor second = make_values({4.0f, 5.0f}, 1, 2, 1);
    const Tensor combined = make_values({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, 1, 5, 1);
    HierarchicalSummaryState chunked;
    HierarchicalSummaryState single;
    ASSERT_TRUE(chunked.reset(32, 4, 3, 1, 1));
    ASSERT_TRUE(single.reset(32, 4, 3, 1, 1));
    ASSERT_TRUE(chunked.append(first));
    ASSERT_TRUE(chunked.append(second));
    ASSERT_TRUE(single.append(combined));

    Tensor chunked_snapshot;
    Tensor single_snapshot;
    ASSERT_TRUE(chunked.snapshot(chunked_snapshot));
    ASSERT_TRUE(single.snapshot(single_snapshot));
    ASSERT_EQ(chunked_snapshot.shape(), single_snapshot.shape());
    ASSERT_EQ(chunked_snapshot.size(), single_snapshot.size());
    for (std::size_t index = 0; index < chunked_snapshot.size(); ++index) {
        EXPECT_FLOAT_EQ(chunked_snapshot.data()[index], single_snapshot.data()[index]) << index;
    }
}

TEST(HierarchicalSummaryTest, UsesBoundedStateForBillionTokenLogicalContext) {
    HierarchicalSummaryState state;
    ASSERT_TRUE(state.reset(1'000'000'000, 8, 16, 2, 4));
    EXPECT_EQ(state.logical_context_length(), 1'000'000'000u);
    EXPECT_EQ(state.state_bytes(), 8u * 2u * 4u * sizeof(double) +
                                  8u * 2u * sizeof(std::size_t) +
                                  8u * 2u * 4u * sizeof(float) + 4u * sizeof(float));
    EXPECT_EQ(state.tokens_processed(), 0u);

    const Tensor chunk = make_values(std::vector<float>(2u * 32u * 4u, 1.0f), 2, 32, 4);
    ASSERT_TRUE(state.append(chunk));
    EXPECT_EQ(state.tokens_processed(), 32u);
}

TEST(HierarchicalSummaryTest, RejectsInvalidInputsAndContextOverflow) {
    HierarchicalSummaryState state;
    std::string error;
    EXPECT_FALSE(state.reset(0, 2, 2, 1, 1, &error));
    EXPECT_FALSE(state.reset(8, 2, 1, 1, 1, &error));
    ASSERT_TRUE(state.reset(2, 2, 2, 1, 1, &error));

    Tensor wrong_shape;
    ASSERT_TRUE(wrong_shape.reset({2, 1, 1}));
    EXPECT_FALSE(state.append(wrong_shape, &error));
    EXPECT_NE(error.find("batch"), std::string::npos);

    const Tensor full = make_values({1.0f, 2.0f}, 1, 2, 1);
    ASSERT_TRUE(state.append(full, &error)) << error;
    const Tensor overflow = make_values({3.0f}, 1, 1, 1);
    EXPECT_FALSE(state.append(overflow, &error));
    EXPECT_NE(error.find("context"), std::string::npos);

    Tensor nonfinite = make_values({std::numeric_limits<float>::infinity()}, 1, 1, 1);
    ASSERT_TRUE(state.reset(8, 2, 2, 1, 1));
    EXPECT_FALSE(state.append(nonfinite, &error));
}

} // namespace
} // namespace attention
