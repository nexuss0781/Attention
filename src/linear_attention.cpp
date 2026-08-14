#include "attention/linear_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace attention {
namespace {

constexpr float kFeatureClipMin = -20.0f;
constexpr float kFeatureClipMax = 20.0f;

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_tensor(const Tensor& tensor) {
    return tensor.valid() && tensor.rank() == 3 &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor;
}

float positive_feature(float value) {
    const float clipped = std::clamp(value, kFeatureClipMin, kFeatureClipMax);
    return std::exp(clipped);
}

} // namespace

bool LinearCausalAttention::reset(std::size_t context_length,
                                  std::size_t hidden_size,
                                  float epsilon,
                                  std::string* error) noexcept {
    if (context_length == 0 || hidden_size == 0) {
        set_error(error, "context length and hidden size must be positive");
        return false;
    }
    if (!std::isfinite(epsilon) || !(epsilon > 0.0f)) {
        set_error(error, "epsilon must be finite and positive");
        return false;
    }
    if (!mask_.reset(context_length, error)) return false;
    context_length_ = context_length;
    hidden_size_ = hidden_size;
    epsilon_ = epsilon;
    if (error != nullptr) error->clear();
    return true;
}

bool LinearCausalAttention::forward(const Tensor& query,
                                    const Tensor& key,
                                    const Tensor& value,
                                    Tensor& output,
                                    std::string* error) const {
    if (context_length_ == 0 || hidden_size_ == 0) {
        set_error(error, "linear causal attention is not initialized");
        return false;
    }
    if (!valid_tensor(query) || !valid_tensor(key) || !valid_tensor(value)) {
        set_error(error, "linear attention inputs must be valid F32 CPU row-major rank-3 tensors");
        return false;
    }
    if (query.shape() != key.shape() || query.shape() != value.shape()) {
        set_error(error, "linear attention input shapes must match");
        return false;
    }
    const auto& shape = query.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    if (!mask_.valid_sequence_length(sequence_length) || shape[2] != hidden_size_) {
        set_error(error, "linear attention input shape exceeds context or hidden size");
        return false;
    }
    if (!query.all_finite() || !key.all_finite() || !value.all_finite()) {
        set_error(error, "linear attention inputs contain NaN or infinity");
        return false;
    }
    if (hidden_size_ > std::numeric_limits<std::size_t>::max() / hidden_size_) {
        set_error(error, "linear attention state size overflows size_t");
        return false;
    }
    if (!output.reset(shape, TensorDataType::F32, TensorDevice::CPU, error)) return false;

    const std::size_t state_size = hidden_size_ * hidden_size_;
    std::vector<double> state(state_size, 0.0);
    std::vector<double> normalizer(hidden_size_, 0.0);
    std::vector<float> query_features(hidden_size_);
    std::vector<float> key_features(hidden_size_);

    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        std::fill(state.begin(), state.end(), 0.0);
        std::fill(normalizer.begin(), normalizer.end(), 0.0);
        for (std::size_t position = 0; position < sequence_length; ++position) {
            const std::size_t row_offset = (batch * sequence_length + position) * hidden_size_;
            for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                key_features[channel] = positive_feature(key.data()[row_offset + channel]);
                normalizer[channel] += static_cast<double>(key_features[channel]);
            }
            for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                const double key_value = static_cast<double>(key_features[key_channel]);
                double* state_row = state.data() + key_channel * hidden_size_;
                for (std::size_t value_channel = 0; value_channel < hidden_size_; ++value_channel) {
                    state_row[value_channel] += key_value *
                        static_cast<double>(value.data()[row_offset + value_channel]);
                }
            }
            for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                query_features[channel] = positive_feature(query.data()[row_offset + channel]);
            }
            double denominator = 0.0;
            for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                denominator += static_cast<double>(query_features[key_channel]) * normalizer[key_channel];
            }
            const double safe_denominator = std::max(denominator, static_cast<double>(epsilon_));
            for (std::size_t value_channel = 0; value_channel < hidden_size_; ++value_channel) {
                double numerator = 0.0;
                for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                    numerator += static_cast<double>(query_features[key_channel]) *
                        state[key_channel * hidden_size_ + value_channel];
                }
                output.data()[row_offset + value_channel] =
                    static_cast<float>(numerator / safe_denominator);
            }
        }
    }
    if (!output.all_finite()) {
        set_error(error, "linear attention output is not finite");
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t LinearCausalAttention::context_length() const noexcept {
    return context_length_;
}

std::size_t LinearCausalAttention::hidden_size() const noexcept {
    return hidden_size_;
}

std::size_t LinearCausalAttention::state_bytes(std::size_t batch_size) const noexcept {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (batch_size == 0 || hidden_size_ == 0 ||
        hidden_size_ > maximum / hidden_size_) {
        return 0;
    }
    const std::size_t state_elements = hidden_size_ * hidden_size_;
    if (state_elements > maximum / sizeof(double) ||
        hidden_size_ > maximum / sizeof(double)) {
        return 0;
    }
    const std::size_t state_bytes = state_elements * sizeof(double);
    const std::size_t normalizer_bytes = hidden_size_ * sizeof(double);
    if (state_bytes > maximum - normalizer_bytes) return 0;
    const std::size_t bytes_per_batch = state_bytes + normalizer_bytes;
    if (batch_size > maximum / bytes_per_batch) return 0;
    return batch_size * bytes_per_batch;
}

} // namespace attention
