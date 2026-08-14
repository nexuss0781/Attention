#include <gtest/gtest.h>

#include "attention/residual_connection.h"

#include <limits>
#include <string>
#include <vector>

namespace attention {
namespace {

TEST(ResidualConnectionTest, AddsCompatibleActivationsOutOfPlace) {
    Tensor main;
    Tensor residual;
    ASSERT_TRUE(main.reset({2, 3, 2}));
    ASSERT_TRUE(residual.reset({2, 3, 2}));
    for (std::size_t index = 0; index < main.size(); ++index) {
        main.data()[index] = static_cast<float>(index);
        residual.data()[index] = static_cast<float>(index + 1);
    }
    ResidualConnection connection;
    Tensor output;
    std::string error;
    ASSERT_TRUE(connection.add(main, residual, output, &error)) << error;
    EXPECT_EQ(output.shape(), (std::vector<std::size_t>{2, 3, 2}));
    for (std::size_t index = 0; index < output.size(); ++index) {
        EXPECT_FLOAT_EQ(output.data()[index], static_cast<float>(2 * index + 1));
    }
}

TEST(ResidualConnectionTest, InPlaceAdditionUsesExistingTargetStorage) {
    Tensor target;
    Tensor residual;
    ASSERT_TRUE(target.reset({1, 2, 3}));
    ASSERT_TRUE(residual.reset({1, 2, 3}));
    target.fill(2.0f);
    residual.fill(0.5f);
    float* original_storage = target.data();
    ResidualConnection connection;
    ASSERT_TRUE(connection.add_in_place(target, residual));
    EXPECT_EQ(target.data(), original_storage);
    for (std::size_t index = 0; index < target.size(); ++index) {
        EXPECT_FLOAT_EQ(target.data()[index], 2.5f);
    }
}

TEST(ResidualConnectionTest, RejectsShapeNonfiniteAndOverflowInputs) {
    Tensor main;
    Tensor residual;
    ASSERT_TRUE(main.reset({1, 1, 2}));
    ASSERT_TRUE(residual.reset({1, 1, 3}));
    ResidualConnection connection;
    Tensor output;
    std::string error;
    EXPECT_FALSE(connection.add(main, residual, output, &error));
    EXPECT_EQ(error, "residual inputs must be finite compatible rank-3 tensors");

    ASSERT_TRUE(residual.reset({1, 1, 2}));
    residual.data()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(connection.add(main, residual, output, &error));
    EXPECT_EQ(error, "residual inputs must be finite compatible rank-3 tensors");

    residual.fill(std::numeric_limits<float>::max());
    main.fill(std::numeric_limits<float>::max());
    EXPECT_FALSE(connection.add(main, residual, output, &error));
    EXPECT_EQ(error, "residual sum contains NaN or infinity");
}

TEST(ResidualConnectionTest, OutOfPlaceOutputMustBeDistinct) {
    Tensor main;
    Tensor residual;
    ASSERT_TRUE(main.reset({1, 1, 2}));
    ASSERT_TRUE(residual.reset({1, 1, 2}));
    ResidualConnection connection;
    std::string error;
    EXPECT_FALSE(connection.add(main, residual, main, &error));
    EXPECT_EQ(error, "out-of-place residual output must be distinct from inputs");
}

} // namespace
} // namespace attention
