#include "attention/transformer_block.h"

#include <limits>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_input(const Tensor& input) {
    return input.valid() && input.rank() == 3 &&
           input.data_type() == TensorDataType::F32 &&
           input.device() == TensorDevice::CPU &&
           input.layout() == TensorLayout::RowMajor &&
           input.all_finite();
}

bool multiply_fits(std::size_t left, std::size_t right) noexcept {
    return left == 0 || right <= std::numeric_limits<std::size_t>::max() / left;
}

} // namespace

TransformerBlock::TransformerBlock(std::size_t layer_index)
    : layer_index_(layer_index),
      attention_normalization_("layers." + std::to_string(layer_index) + ".norm1"),
      qkv_projection_(),
      attention_(),
      feed_forward_normalization_("layers." + std::to_string(layer_index) + ".norm2"),
      feed_forward_("layers." + std::to_string(layer_index) + ".ffn") {}

bool TransformerBlock::register_parameters(const TransformerConfig& config,
                                           ParameterStore& parameters,
                                           std::string* error) {
    if (!config.validate(error)) return false;
    if (config.layer_count == 0 || layer_index_ >= config.layer_count) {
        set_error(error, "transformer block layer index is outside the configured layer count");
        return false;
    }
    if (!config.causal) {
        set_error(error, "TransformerBlock requires a causal configuration");
        return false;
    }
    if (initialized_) {
        set_error(error, "transformer block is already initialized");
        return false;
    }

    if (!attention_normalization_.register_parameters(config, parameters, error) ||
        !qkv_projection_.register_parameters(config, layer_index_, parameters, error) ||
        !attention_.reset(config.context_length, config.hidden_size, 1e-6f, error) ||
        !feed_forward_normalization_.register_parameters(config, parameters, error) ||
        !feed_forward_.register_parameters(config, parameters, error)) {
        return false;
    }
    hidden_size_ = config.hidden_size;
    context_length_ = config.context_length;
    initialized_ = true;
    if (error != nullptr) error->clear();
    return true;
}

bool TransformerBlock::forward(const Tensor& input,
                               const ParameterStore& parameters,
                               Tensor& output,
                               std::string* error) const {
    if (!initialized_ || hidden_size_ == 0 || context_length_ == 0) {
        set_error(error, "transformer block is not initialized");
        return false;
    }
    if (!valid_input(input)) {
        set_error(error, "transformer block input must be a finite F32 CPU row-major rank-3 tensor");
        return false;
    }
    const auto& shape = input.shape();
    if (shape[0] == 0 || shape[1] == 0 || shape[2] != hidden_size_ || shape[1] > context_length_ ||
        !multiply_fits(shape[0], shape[1]) ||
        !multiply_fits(shape[0] * shape[1], shape[2])) {
        set_error(error, "transformer block input shape does not match configured dimensions");
        return false;
    }

    Tensor normalized_attention;
    QKVOutput projected;
    Tensor attended;
    Tensor after_attention;
    Tensor normalized_feed_forward;
    Tensor feed_forward_output;

    if (!attention_normalization_.forward(input, parameters, normalized_attention, error)) return false;
    if (!qkv_projection_.forward(normalized_attention, parameters, projected, error)) return false;
    if (!attention_.forward(projected.query, projected.key, projected.value, attended, error)) return false;
    if (!residual_.add(input, attended, after_attention, error)) return false;
    if (!feed_forward_normalization_.forward(after_attention, parameters, normalized_feed_forward, error)) return false;
    if (!feed_forward_.forward(normalized_feed_forward, parameters, feed_forward_output, error)) return false;
    if (!residual_.add(after_attention, feed_forward_output, output, error)) return false;
    if (error != nullptr) error->clear();
    return true;
}

std::size_t TransformerBlock::layer_index() const noexcept { return layer_index_; }
std::size_t TransformerBlock::hidden_size() const noexcept { return hidden_size_; }
std::size_t TransformerBlock::context_length() const noexcept { return context_length_; }
bool TransformerBlock::initialized() const noexcept { return initialized_; }

} // namespace attention
