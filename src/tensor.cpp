#include "attention/tensor.h"

#include <cmath>
#include <limits>
#include <utility>

namespace attention {
namespace {

bool multiply_fits(std::size_t a, std::size_t b) noexcept {
    return a == 0 || b <= std::numeric_limits<std::size_t>::max() / a;
}

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

} // namespace

bool Tensor::reset(std::vector<std::size_t> shape,
                   TensorDataType data_type,
                   TensorDevice device,
                   std::string* error) {
    if (shape.empty()) {
        set_error(error, "tensor shape must not be empty");
        return false;
    }
    if (data_type != TensorDataType::F32) {
        set_error(error, "only F32 tensors are implemented");
        return false;
    }
    if (device != TensorDevice::CPU) {
        set_error(error, "only CPU tensors are implemented");
        return false;
    }

    std::size_t element_count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            set_error(error, "tensor dimensions must be positive");
            return false;
        }
        if (!multiply_fits(element_count, dimension)) {
            set_error(error, "tensor element count overflows size_t");
            return false;
        }
        element_count *= dimension;
    }
    if (!multiply_fits(element_count, sizeof(float))) {
        set_error(error, "tensor byte count overflows size_t");
        return false;
    }

    std::vector<std::size_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        if (!multiply_fits(strides[index], shape[index])) {
            set_error(error, "tensor stride overflows size_t");
            return false;
        }
        strides[index - 1] = strides[index] * shape[index];
    }

    try {
        std::vector<float> data(element_count, 0.0f);
        shape_ = std::move(shape);
        strides_ = std::move(strides);
        data_ = std::move(data);
        data_type_ = data_type;
        device_ = device;
        layout_ = TensorLayout::RowMajor;
    } catch (...) {
        set_error(error, "tensor storage allocation failed");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool Tensor::valid() const noexcept {
    return !shape_.empty() && shape_.size() == strides_.size() &&
           !data_.empty() && data_.size() == size();
}

std::size_t Tensor::rank() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    if (shape_.empty()) return 0;
    std::size_t count = 1;
    for (const std::size_t dimension : shape_) {
        if (!multiply_fits(count, dimension)) return 0;
        count *= dimension;
    }
    return count;
}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

const std::vector<std::size_t>& Tensor::strides() const noexcept {
    return strides_;
}

TensorDataType Tensor::data_type() const noexcept {
    return data_type_;
}

TensorDevice Tensor::device() const noexcept {
    return device_;
}

TensorLayout Tensor::layout() const noexcept {
    return layout_;
}

float* Tensor::data() noexcept {
    return data_.data();
}

const float* Tensor::data() const noexcept {
    return data_.data();
}

bool Tensor::all_finite() const noexcept {
    for (const float value : data_) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

void Tensor::fill(float value) noexcept {
    for (float& element : data_) element = value;
}

} // namespace attention
