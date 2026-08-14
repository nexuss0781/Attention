#include "attention/training_data.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace attention {
namespace {

TEST(TrainingBatchLoaderTest, PacksDeterministicFullBatchesAndResets) {
    TrainingBatchLoader loader;
    std::string error;
    ASSERT_TRUE(loader.initialize({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 2, 3, true, &error)) << error;
    EXPECT_EQ(loader.batch_count(), 1U);
    EXPECT_FALSE(loader.exhausted());

    TrainingBatch batch;
    ASSERT_TRUE(loader.next(batch, &error)) << error;
    EXPECT_EQ(batch.batch_size, 2U);
    EXPECT_EQ(batch.sequence_length, 3U);
    EXPECT_EQ(batch.token_offset, 0U);
    EXPECT_EQ(batch.token_ids, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(loader.tokens_processed(), 6U);
    EXPECT_TRUE(loader.exhausted());
    EXPECT_FALSE(loader.next(batch, &error));

    loader.reset();
    ASSERT_TRUE(loader.next(batch, &error)) << error;
    EXPECT_EQ(batch.token_ids, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
}

TEST(TrainingBatchLoaderTest, EmitsCausalPartialBatchWhenConfigured) {
    TrainingBatchLoader loader;
    std::string error;
    ASSERT_TRUE(loader.initialize({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 2, 3, false, &error)) << error;
    EXPECT_EQ(loader.batch_count(), 2U);

    TrainingBatch batch;
    ASSERT_TRUE(loader.next(batch, &error)) << error;
    EXPECT_EQ(batch.batch_size, 2U);
    EXPECT_EQ(batch.sequence_length, 3U);
    ASSERT_TRUE(loader.next(batch, &error)) << error;
    EXPECT_EQ(batch.batch_size, 1U);
    EXPECT_EQ(batch.sequence_length, 4U);
    EXPECT_EQ(batch.token_offset, 6U);
    EXPECT_EQ(batch.token_ids, (std::vector<std::size_t>{6, 7, 8, 9}));
    EXPECT_TRUE(loader.exhausted());
}

TEST(TrainingBatchLoaderTest, RejectsInvalidShapesAndUnusableRemainders) {
    TrainingBatchLoader loader;
    std::string error;
    EXPECT_FALSE(loader.initialize({0, 1, 2, 3}, 0, 4, true, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(loader.initialize({0, 1, 2, 3}, 1, 1, true, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(loader.initialize({0}, 1, 2, true, &error));
    EXPECT_FALSE(error.empty());
    error.clear();
    EXPECT_FALSE(loader.initialize({0, 1, 2, 3, 4, 5, 6}, 2, 3, false, &error));
    EXPECT_FALSE(error.empty());

    ASSERT_TRUE(loader.initialize({0, 1, 2, 3, 4, 5}, 2, 3, true, &error)) << error;
    error.clear();
    EXPECT_FALSE(loader.initialize({0, 1, 2, 3, 4, 5, 6}, 2, 3, false, &error));
    EXPECT_FALSE(error.empty());
    TrainingBatch batch;
    ASSERT_TRUE(loader.next(batch, &error)) << error;
    EXPECT_EQ(batch.token_ids, (std::vector<std::size_t>{0, 1, 2, 3, 4, 5}));
}

TEST(TrainingBatchLoaderTest, RejectsNextBeforeInitialization) {
    TrainingBatchLoader loader;
    TrainingBatch batch;
    std::string error;
    EXPECT_TRUE(loader.exhausted());
    EXPECT_FALSE(loader.next(batch, &error));
    EXPECT_FALSE(error.empty());
}

} // namespace
} // namespace attention
