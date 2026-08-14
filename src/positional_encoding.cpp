#include "attention/positional_encoding.h"

#include <cmath>
#include <limits>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

} // namespace

bool SinusoidalPositionEncoding::reset(std::size_t context_length,
                                       std::size_t hidden_size,
                                       std::string* error) noexcept {
    if (context_length == 0 || hidden_size == 0) {
        set_error(error, "context length and hidden size must be positive");
        return false;
    }
    context_length_ = context_length;
    hidden_size_ = hidden_size;
    if (error != nullptr) error->clear();
    return true;
}

bool SinusoidalPositionEncoding::apply(const Tensor& input,
                                       Tensor& output,
                                       std::string* error) const {
    return apply_at(input, 0, output, error);
}

bool SinusoidalPositionEncoding::apply_at(const Tensor& input,
                                          std::size_t position_offset,
                                          Tensor& output,
                                          std::string* error) const {
    if (context_length_ == 0 || hidden_size_ == 0) {
        set_error(error, "positional encoding is not initialized");
        return false;
    }
    if (!input.valid() || input.rank() != 3 ||
        input.data_type() != TensorDataType::F32 ||
        input.device() != TensorDevice::CPU ||
        input.layout() != TensorLayout::RowMajor) {
        set_error(error, "positional encoding input must be a valid F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    if (batch_size == 0 || sequence_length == 0 || shape[2] != hidden_size_) {
        set_error(error, "positional encoding input shape does not match hidden size");
        return false;
    }
    if (position_offset > context_length_ ||
        sequence_length > context_length_ - position_offset) {
        set_error(error, "absolute positional range exceeds positional-encoding context length");
        return false;
    }
    if (!input.all_finite()) {
        set_error(error, "positional encoding input contains NaN or infinity");
        return false;
    }
    if (!output.reset(shape, TensorDataType::F32, TensorDevice::CPU, error)) return false;

    constexpr double log_base = 9.210340371976184;
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t row_offset = (batch * sequence_length + position) * hidden_size_;
            const double position_value = static_cast<double>(position_offset + position);
            for (std::size_t dimension = 0; dimension < hidden_size_; ++dimension) {
                const std::size_t pair_index = dimension / 2;
                const double exponent = (2.0 * static_cast<double>(pair_index)) /
                    static_cast<double>(hidden_size_);
                const double angle = position_value * std::exp(-log_base * exponent);
                const float positional_value = (dimension % 2 == 0)
                    ? static_cast<float>(std::sin(angle))
                    : static_cast<float>(std::cos(angle));
                output.data()[row_offset + dimension] = input.data()[row_offset + dimension] + positional_value;
            }
        }
    }
    if (!output.all_finite()) {
        set_error(error, "positional encoding output is not finite");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t SinusoidalPositionEncoding::context_length() const noexcept {
    return context_length_;
}

std::size_t SinusoidalPositionEncoding::hidden_size() const noexcept {
    return hidden_size_;
}

} // namespace attention
