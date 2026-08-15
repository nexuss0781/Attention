#include "attention/transformer_model.h"
#include "analytical_backward.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool multiply_fits(std::size_t left, std::size_t right) noexcept {
    return left == 0 || right <= std::numeric_limits<std::size_t>::max() / left;
}

bool valid_logits(const Tensor& logits) {
    return logits.valid() && logits.rank() == 3 &&
           logits.data_type() == TensorDataType::F32 &&
           logits.device() == TensorDevice::CPU &&
           logits.layout() == TensorLayout::RowMajor &&
           logits.all_finite();
}

} // namespace

bool TransformerModel::register_parameters(const TransformerConfig& config,
                                            ParameterStore& parameters,
                                            std::string* error) {
    if (!config.validate(error)) return false;
    if (initialized_) {
        set_error(error, "transformer model is already initialized");
        return false;
    }
    if (!embedding_.register_parameters(config, parameters, error)) return false;
    if (!positional_encoding_.reset(config.context_length, config.hidden_size, error)) return false;
    blocks_.clear();
    try {
        blocks_.reserve(config.layer_count);
        for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
            blocks_.emplace_back(layer);
            if (!blocks_.back().register_parameters(config, parameters, error)) {
                blocks_.clear();
                return false;
            }
        }
    } catch (const std::bad_alloc&) {
        blocks_.clear();
        set_error(error, "transformer block allocation failed");
        return false;
    }
    if (!final_output_.register_parameters(config, parameters, error) ||
        !output_projection_.register_parameters(config, parameters, error)) {
        blocks_.clear();
        return false;
    }
    config_ = config;
    initialized_ = true;
    if (error != nullptr) error->clear();
    return true;
}

bool TransformerModel::forward(const std::vector<std::size_t>& token_ids,
                               std::size_t batch_size,
                               std::size_t sequence_length,
                               const ParameterStore& parameters,
                               Tensor& logits,
                               std::string* error) const {
    if (!initialized_) {
        set_error(error, "transformer model is not initialized");
        return false;
    }
    if (batch_size == 0 || sequence_length == 0 || sequence_length > config_.context_length ||
        !multiply_fits(batch_size, sequence_length) ||
        token_ids.size() != batch_size * sequence_length) {
        set_error(error, "transformer model token shape does not match configured dimensions");
        return false;
    }

    Tensor hidden;
    if (!embedding_.forward(token_ids, batch_size, sequence_length, parameters, hidden, error)) return false;
    Tensor positioned;
    if (!positional_encoding_.apply(hidden, positioned, error)) return false;
    hidden = std::move(positioned);
    for (const TransformerBlock& block : blocks_) {
        Tensor next;
        if (!block.forward(hidden, parameters, next, error)) return false;
        hidden = std::move(next);
    }
    Tensor normalized;
    if (!final_output_.forward(hidden, parameters, normalized, error)) return false;
    if (!output_projection_.forward(normalized, parameters, logits, error)) return false;
    if (error != nullptr) error->clear();
    return true;
}

bool TransformerModel::causal_loss(const std::vector<std::size_t>& token_ids,
                                   std::size_t batch_size,
                                   std::size_t sequence_length,
                                   const ParameterStore& parameters,
                                   float& loss,
                                   std::string* error) const {
    Tensor logits;
    if (!forward(token_ids, batch_size, sequence_length, parameters, logits, error)) return false;
    return causal_cross_entropy(logits, token_ids, loss, error);
}

bool TransformerModel::backward(const std::vector<std::size_t>& token_ids,
                                std::size_t batch_size,
                                std::size_t sequence_length,
                                ParameterStore& parameters,
                                float difference_step,
                                std::string* error) const {
    if (!initialized_) {
        set_error(error, "transformer model is not initialized");
        return false;
    }
    if (!std::isfinite(difference_step) || !(difference_step > 0.0f)) {
        set_error(error, "backward difference step must be finite and positive");
        return false;
    }
    return analytical_backward(*this, token_ids, batch_size, sequence_length, parameters, error);
}

bool TransformerModel::causal_cross_entropy(const Tensor& logits,
                                            const std::vector<std::size_t>& target_token_ids,
                                            float& loss,
                                            std::string* error) {
    if (!valid_logits(logits)) {
        set_error(error, "causal-loss logits must be finite F32 CPU row-major rank-3 data");
        return false;
    }
    const auto& shape = logits.shape();
    const std::size_t batch_size = shape[0];
    const std::size_t sequence_length = shape[1];
    const std::size_t vocabulary_size = shape[2];
    if (batch_size == 0 || sequence_length < 2 || vocabulary_size == 0 ||
        !multiply_fits(batch_size, sequence_length) ||
        target_token_ids.size() != batch_size * sequence_length) {
        set_error(error, "causal-loss target shape requires matching batch and sequence with length at least two");
        return false;
    }

    double total_loss = 0.0;
    const std::size_t prediction_count = batch_size * (sequence_length - 1);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        for (std::size_t position = 0; position + 1 < sequence_length; ++position) {
            const std::size_t target = target_token_ids[batch * sequence_length + position + 1];
            if (target >= vocabulary_size) {
                set_error(error, "causal-loss target token is outside vocabulary");
                return false;
            }
            const std::size_t row_offset = (batch * sequence_length + position) * vocabulary_size;
            const float* row = logits.data() + row_offset;
            float maximum = row[0];
            for (std::size_t vocabulary = 1; vocabulary < vocabulary_size; ++vocabulary) {
                maximum = std::max(maximum, row[vocabulary]);
            }
            double exponential_sum = 0.0;
            for (std::size_t vocabulary = 0; vocabulary < vocabulary_size; ++vocabulary) {
                exponential_sum += std::exp(static_cast<double>(row[vocabulary] - maximum));
            }
            if (!(exponential_sum > 0.0) || !std::isfinite(exponential_sum)) {
                set_error(error, "causal-loss softmax normalization is not finite");
                return false;
            }
            total_loss += std::log(exponential_sum) + static_cast<double>(maximum) -
                          static_cast<double>(row[target]);
        }
    }
    total_loss /= static_cast<double>(prediction_count);
    if (!std::isfinite(total_loss) || total_loss < 0.0 ||
        total_loss > static_cast<double>(std::numeric_limits<float>::max())) {
        set_error(error, "causal-loss result is not finite");
        return false;
    }
    loss = static_cast<float>(total_loss);
    if (error != nullptr) error->clear();
    return true;
}

const TransformerConfig& TransformerModel::config() const noexcept {
    return config_;
}

std::size_t TransformerModel::vocabulary_size() const noexcept {
 return config_.vocabulary_size; }
std::size_t TransformerModel::context_length() const noexcept { return config_.context_length; }
std::size_t TransformerModel::hidden_size() const noexcept { return config_.hidden_size; }
std::size_t TransformerModel::layer_count() const noexcept { return config_.layer_count; }
bool TransformerModel::initialized() const noexcept { return initialized_; }

} // namespace attention
