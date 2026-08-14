#include "attention/linear_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

namespace attention {
namespace {

constexpr float kFeatureClipMin = -20.0f;
constexpr float kFeatureClipMax = 20.0f;

void set_stream_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_stream_tensor(const Tensor& tensor) {
    return tensor.valid() && tensor.rank() == 3 &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor;
}

float stream_feature(float value) {
    return std::exp(std::clamp(value, kFeatureClipMin, kFeatureClipMax));
}

} // namespace

bool LinearAttentionState::reset(std::size_t context_length,
                                 std::size_t batch_size,
                                 std::size_t hidden_size,
                                 float epsilon,
                                 std::string* error) noexcept {
    if (context_length == 0 || batch_size == 0 || hidden_size == 0) {
        set_stream_error(error, "stream context, batch size, and hidden size must be positive");
        return false;
    }
    if (!std::isfinite(epsilon) || !(epsilon > 0.0f)) {
        set_stream_error(error, "stream epsilon must be finite and positive");
        return false;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (hidden_size > maximum / hidden_size) {
        set_stream_error(error, "stream state size overflows size_t");
        return false;
    }
    const std::size_t state_per_batch = hidden_size * hidden_size;
    if (batch_size > maximum / state_per_batch) {
        set_stream_error(error, "stream batch state size overflows size_t");
        return false;
    }
    try {
        state_.assign(batch_size * state_per_batch, 0.0);
        normalizer_.assign(batch_size * hidden_size, 0.0);
        query_features_.assign(hidden_size, 0.0f);
        key_features_.assign(hidden_size, 0.0f);
    } catch (const std::bad_alloc&) {
        clear();
        set_stream_error(error, "stream state allocation failed");
        return false;
    }
    context_length_ = context_length;
    batch_size_ = batch_size;
    hidden_size_ = hidden_size;
    tokens_processed_ = 0;
    epsilon_ = epsilon;
    if (error != nullptr) error->clear();
    return true;
}

bool LinearAttentionState::append(const Tensor& query,
                                  const Tensor& key,
                                  const Tensor& value,
                                  Tensor& output,
                                  std::string* error) {
    if (context_length_ == 0 || batch_size_ == 0 || hidden_size_ == 0) {
        set_stream_error(error, "linear attention stream is not initialized");
        return false;
    }
    if (!valid_stream_tensor(query) || !valid_stream_tensor(key) ||
        !valid_stream_tensor(value)) {
        set_stream_error(error, "stream inputs must be valid F32 CPU row-major rank-3 tensors");
        return false;
    }
    if (query.shape() != key.shape() || query.shape() != value.shape()) {
        set_stream_error(error, "stream input shapes must match");
        return false;
    }
    const auto& shape = query.shape();
    const std::size_t chunk_batch = shape[0];
    const std::size_t chunk_length = shape[1];
    if (chunk_batch != batch_size_ || chunk_length == 0 || shape[2] != hidden_size_ ||
        tokens_processed_ > context_length_ ||
        chunk_length > context_length_ - tokens_processed_) {
        set_stream_error(error, "stream chunk exceeds configured batch, hidden size, or context");
        return false;
    }
    if (!query.all_finite() || !key.all_finite() || !value.all_finite()) {
        set_stream_error(error, "stream inputs contain NaN or infinity");
        return false;
    }
    if (!output.reset(shape, TensorDataType::F32, TensorDevice::CPU, error)) return false;

    const std::size_t state_per_batch = hidden_size_ * hidden_size_;
    for (std::size_t batch = 0; batch < batch_size_; ++batch) {
        double* state = state_.data() + batch * state_per_batch;
        double* normalizer = normalizer_.data() + batch * hidden_size_;
        for (std::size_t position = 0; position < chunk_length; ++position) {
            const std::size_t row_offset = (batch * chunk_length + position) * hidden_size_;
            for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                key_features_[channel] = stream_feature(key.data()[row_offset + channel]);
                normalizer[channel] += static_cast<double>(key_features_[channel]);
            }
            for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                const double key_value = static_cast<double>(key_features_[key_channel]);
                double* state_row = state + key_channel * hidden_size_;
                for (std::size_t value_channel = 0; value_channel < hidden_size_; ++value_channel) {
                    state_row[value_channel] += key_value *
                        static_cast<double>(value.data()[row_offset + value_channel]);
                }
            }
            for (std::size_t channel = 0; channel < hidden_size_; ++channel) {
                query_features_[channel] = stream_feature(query.data()[row_offset + channel]);
            }
            double denominator = 0.0;
            for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                denominator += static_cast<double>(query_features_[key_channel]) * normalizer[key_channel];
            }
            const double safe_denominator = std::max(denominator, static_cast<double>(epsilon_));
            for (std::size_t value_channel = 0; value_channel < hidden_size_; ++value_channel) {
                double numerator = 0.0;
                for (std::size_t key_channel = 0; key_channel < hidden_size_; ++key_channel) {
                    numerator += static_cast<double>(query_features_[key_channel]) *
                        state[key_channel * hidden_size_ + value_channel];
                }
                output.data()[row_offset + value_channel] =
                    static_cast<float>(numerator / safe_denominator);
            }
        }
    }
    if (!output.all_finite()) {
        set_stream_error(error, "stream output is not finite");
        return false;
    }
    tokens_processed_ += chunk_length;
    if (error != nullptr) error->clear();
    return true;
}

void LinearAttentionState::clear() noexcept {
    context_length_ = 0;
    batch_size_ = 0;
    hidden_size_ = 0;
    tokens_processed_ = 0;
    epsilon_ = 1e-6f;
    state_.clear();
    normalizer_.clear();
    query_features_.clear();
    key_features_.clear();
}

std::size_t LinearAttentionState::context_length() const noexcept {
    return context_length_;
}

std::size_t LinearAttentionState::batch_size() const noexcept {
    return batch_size_;
}

std::size_t LinearAttentionState::hidden_size() const noexcept {
    return hidden_size_;
}

std::size_t LinearAttentionState::tokens_processed() const noexcept {
    return tokens_processed_;
}

std::size_t LinearAttentionState::state_bytes() const noexcept {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t state_bytes = state_.size() > maximum / sizeof(double)
        ? 0 : state_.size() * sizeof(double);
    const std::size_t normalizer_bytes = normalizer_.size() > maximum / sizeof(double)
        ? 0 : normalizer_.size() * sizeof(double);
    const std::size_t query_bytes = query_features_.size() > maximum / sizeof(float)
        ? 0 : query_features_.size() * sizeof(float);
    const std::size_t key_bytes = key_features_.size() > maximum / sizeof(float)
        ? 0 : key_features_.size() * sizeof(float);
    if (state_bytes == 0 || normalizer_bytes > maximum - state_bytes ||
        query_bytes > maximum - state_bytes - normalizer_bytes ||
        key_bytes > maximum - state_bytes - normalizer_bytes - query_bytes) {
        return 0;
    }
    return state_bytes + normalizer_bytes + query_bytes + key_bytes;
}

} // namespace attention
