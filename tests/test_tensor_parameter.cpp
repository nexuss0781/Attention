#include <gtest/gtest.h>

#include "attention/parameter_store.h"
#include "attention/tensor.h"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace attention {
namespace {

TEST(TensorTest, RowMajorShapeStridesAndMetadata) {
    Tensor tensor;
    std::string error;
    ASSERT_TRUE(tensor.reset({2, 3, 4}, TensorDataType::F32, TensorDevice::CPU, &error)) << error;
    EXPECT_TRUE(error.empty());
    EXPECT_TRUE(tensor.valid());
    EXPECT_EQ(tensor.rank(), 3u);
    EXPECT_EQ(tensor.size(), 24u);
    EXPECT_EQ(tensor.shape(), (std::vector<std::size_t>{2, 3, 4}));
    EXPECT_EQ(tensor.strides(), (std::vector<std::size_t>{12, 4, 1}));
    EXPECT_EQ(tensor.data_type(), TensorDataType::F32);
    EXPECT_EQ(tensor.device(), TensorDevice::CPU);
    EXPECT_EQ(tensor.layout(), TensorLayout::RowMajor);
    EXPECT_TRUE(tensor.all_finite());
    EXPECT_FLOAT_EQ(tensor.data()[0], 0.0f);
}

TEST(TensorTest, RejectsInvalidShapesAndUnsupportedMetadata) {
    Tensor tensor;
    std::string error;
    EXPECT_FALSE(tensor.reset({}, TensorDataType::F32, TensorDevice::CPU, &error));
    EXPECT_EQ(error, "tensor shape must not be empty");
    EXPECT_FALSE(tensor.reset({2, 0}, TensorDataType::F32, TensorDevice::CPU, &error));
    EXPECT_EQ(error, "tensor dimensions must be positive");
    EXPECT_FALSE(tensor.reset({std::numeric_limits<std::size_t>::max(), 2},
                              TensorDataType::F32, TensorDevice::CPU, &error));
    EXPECT_EQ(error, "tensor element count overflows size_t");
}

TEST(TensorTest, OwnsContiguousStorageAndReportsFiniteness) {
    Tensor tensor;
    ASSERT_TRUE(tensor.reset({4}, TensorDataType::F32, TensorDevice::CPU));
    tensor.fill(2.5f);
    EXPECT_TRUE(tensor.all_finite());
    EXPECT_FLOAT_EQ(tensor.data()[3], 2.5f);
    tensor.data()[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(tensor.all_finite());

    Tensor copy = tensor;
    ASSERT_EQ(copy.size(), tensor.size());
    copy.data()[0] = 9.0f;
    EXPECT_FLOAT_EQ(tensor.data()[0], 2.5f);
    Tensor moved = std::move(copy);
    EXPECT_EQ(moved.size(), 4u);
    EXPECT_FLOAT_EQ(moved.data()[0], 9.0f);
}

bool add_standard_parameters(ParameterStore& store) {
    return store.add("layers.1.attention.q_proj.weight", {8, 8}) &&
           store.add("embedding.weight", {32, 8}) &&
           store.add("layers.1.norm.weight", {8});
}

TEST(ParameterStoreTest, NamesAreStableAndShapesMatch) {
    ParameterStore store;
    ASSERT_TRUE(add_standard_parameters(store));
    EXPECT_EQ(store.size(), 3u);
    EXPECT_EQ(store.names(), (std::vector<std::string>{
        "embedding.weight", "layers.1.attention.q_proj.weight", "layers.1.norm.weight"}));
    const Parameter* parameter = store.find("embedding.weight");
    ASSERT_NE(parameter, nullptr);
    EXPECT_EQ(parameter->value.shape(), (std::vector<std::size_t>{32, 8}));
    EXPECT_EQ(parameter->gradient.shape(), parameter->value.shape());
    EXPECT_TRUE(parameter->gradient.all_finite());
}

TEST(ParameterStoreTest, RejectsInvalidAndDuplicateNames) {
    ParameterStore store;
    std::string error;
    EXPECT_FALSE(store.add("", {2, 2}, &error));
    EXPECT_FALSE(store.add("bad name", {2, 2}, &error));
    ASSERT_TRUE(store.add("weight", {2, 2}, &error)) << error;
    EXPECT_FALSE(store.add("weight", {2, 2}, &error));
    EXPECT_EQ(error, "parameter name already exists");
}

TEST(ParameterStoreTest, InitializationIsDeterministicAndGradientsClear) {
    ParameterStore first;
    ParameterStore second;
    ASSERT_TRUE(add_standard_parameters(first));
    ASSERT_TRUE(add_standard_parameters(second));
    ASSERT_TRUE(first.initialize(1234));
    ASSERT_TRUE(second.initialize(1234));
    ASSERT_TRUE(first.all_finite());
    ASSERT_TRUE(second.all_finite());

    for (const std::string& name : first.names()) {
        const Parameter* left = first.find(name);
        const Parameter* right = second.find(name);
        ASSERT_NE(left, nullptr);
        ASSERT_NE(right, nullptr);
        ASSERT_EQ(left->value.size(), right->value.size());
        for (std::size_t index = 0; index < left->value.size(); ++index) {
            EXPECT_FLOAT_EQ(left->value.data()[index], right->value.data()[index]);
            EXPECT_FLOAT_EQ(left->gradient.data()[index], 0.0f);
        }
    }

    Parameter* parameter = first.find("embedding.weight");
    ASSERT_NE(parameter, nullptr);
    parameter->gradient.fill(3.0f);
    ASSERT_TRUE(parameter->gradient.all_finite());
    first.clear_gradients();
    for (std::size_t index = 0; index < parameter->gradient.size(); ++index) {
        EXPECT_FLOAT_EQ(parameter->gradient.data()[index], 0.0f);
    }
}

} // namespace
} // namespace attention
