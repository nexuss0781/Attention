#include "attention/token_embedding.h"

#include <algorithm>
#include <limits>

namespace attention {
namespace {

bool multiply_fits(std::size_t a, std::size_t b) noexcept {
    return a == 0 || b <= std::numeric_limits<std::size_t>::max() / a;
}

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

} // namespace

bool TokenEmbedding::register_parameters(const TransformerConfig& config,
                                         ParameterStore& parameters,
                                         std::string* error) {
    if (!config.validate(error)) return false;
    vocabulary_size_ = config.vocabulary_size;
    hidden_size_ = config.hidden_size;
    if (!parameters.add(parameter_name_, {vocabulary_size_, hidden_size_}, error)) {
        vocabulary_size_ = 0;
        hidden_size_ = 0;
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool TokenEmbedding::forward(const std::vector<std::size_t>& token_ids,
                             std::size_t batch_size,
                             std::size_t sequence_length,
                             const ParameterStore& parameters,
                             Tensor& output,
                             std::string* error) const {
    if (vocabulary_size_ == 0 || hidden_size_ == 0) {
        set_error(error, "token embedding is not initialized");
        return false;
    }
    if (batch_size == 0 || sequence_length == 0) {
        set_error(error, "batch size and sequence length must be positive");
        return false;
    }
    if (!multiply_fits(batch_size, sequence_length) ||
        !multiply_fits(batch_size * sequence_length, hidden_size_)) {
        set_error(error, "embedding output shape overflows size_t");
        return false;
    }
    const std::size_t token_count = batch_size * sequence_length;
    if (token_ids.size() != token_count) {
        set_error(error, "token ID count does not match batch shape");
        return false;
    }
    for (const std::size_t token_id : token_ids) {
        if (token_id >= vocabulary_size_) {
            set_error(error, "token ID is outside the vocabulary");
            return false;
        }
    }

    const Parameter* parameter = parameters.find(parameter_name_);
    if (parameter == nullptr ||
        parameter->value.shape() != std::vector<std::size_t>{vocabulary_size_, hidden_size_}) {
        set_error(error, "embedding parameter is missing or has the wrong shape");
        return false;
    }
    if (!output.reset({batch_size, sequence_length, hidden_size_},
                      TensorDataType::F32, TensorDevice::CPU, error)) {
        return false;
    }
    for (std::size_t index = 0; index < token_count; ++index) {
        const float* source = parameter->value.data() + token_ids[index] * hidden_size_;
        float* destination = output.data() + index * hidden_size_;
        std::copy_n(source, hidden_size_, destination);
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t TokenEmbedding::vocabulary_size() const noexcept {
    return vocabulary_size_;
}

std::size_t TokenEmbedding::hidden_size() const noexcept {
    return hidden_size_;
}

const std::string& TokenEmbedding::parameter_name() const noexcept {
    return parameter_name_;
}

} // namespace attention
