#include "attention/vocabulary_projection.h"

#include <cmath>
#include <limits>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
}

bool valid_hidden(const Tensor& tensor) {
    return tensor.valid() && tensor.rank() == 3 &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor &&
           tensor.all_finite();
}

bool valid_weight(const Tensor& tensor, std::size_t vocabulary, std::size_t hidden) {
    return tensor.valid() && tensor.rank() == 2 &&
           tensor.shape() == std::vector<std::size_t>{vocabulary, hidden} &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor &&
           tensor.all_finite();
}

bool valid_bias(const Tensor& tensor, std::size_t vocabulary) {
    return tensor.valid() && tensor.rank() == 1 &&
           tensor.shape() == std::vector<std::size_t>{vocabulary} &&
           tensor.data_type() == TensorDataType::F32 &&
           tensor.device() == TensorDevice::CPU &&
           tensor.layout() == TensorLayout::RowMajor &&
           tensor.all_finite();
}

} // namespace

bool VocabularyProjection::register_parameters(const TransformerConfig& config,
                                              ParameterStore& parameters,
                                              std::string* error) {
    if (!config.validate(error)) return false;
    if (config.vocabulary_size == 0 || config.hidden_size == 0) {
        set_error(error, "vocabulary projection dimensions must be positive");
        return false;
    }
    const std::string actual_weight = config.tie_embeddings ? "embedding.weight" : weight_name_;
    if (!config.tie_embeddings && (parameters.find(actual_weight) != nullptr || parameters.find(bias_name_) != nullptr)) {
        set_error(error, "vocabulary projection parameter name already exists");
        return false;
    }
    if (config.tie_embeddings) {
        if (parameters.find(actual_weight) == nullptr) {
            set_error(error, "tied vocabulary projection requires embedding.weight");
            return false;
        }
    } else {
        if (parameters.find(actual_weight) != nullptr) {
            set_error(error, "vocabulary projection parameter name already exists");
            return false;
        }
        if (!parameters.add(actual_weight, {config.vocabulary_size, config.hidden_size}, error)) {
            return false;
        }
    }
    if (parameters.find(bias_name_) != nullptr) {
        set_error(error, "vocabulary projection parameter name already exists");
        return false;
    }
    if (!parameters.add(bias_name_, {config.vocabulary_size}, error)) return false;
    vocabulary_size_ = config.vocabulary_size;
    hidden_size_ = config.hidden_size;
    tied_embeddings_ = config.tie_embeddings;
    if (error != nullptr) error->clear();
    return true;
}

bool VocabularyProjection::forward(const Tensor& hidden,
                                  const ParameterStore& parameters,
                                  Tensor& logits,
                                  std::string* error) const {
    if (vocabulary_size_ == 0 || hidden_size_ == 0) {
        set_error(error, "vocabulary projection is not initialized");
        return false;
    }
    if (!valid_hidden(hidden) || hidden.shape()[2] != hidden_size_ ||
        hidden.shape()[0] == 0 || hidden.shape()[1] == 0) {
        set_error(error, "vocabulary projection hidden input is invalid or has the wrong shape");
        return false;
    }
    const std::string actual_weight = tied_embeddings_ ? "embedding.weight" : weight_name_;
    const Parameter* weight = parameters.find(actual_weight);
    const Parameter* bias = parameters.find(bias_name_);
    if (weight == nullptr || bias == nullptr ||
        !valid_weight(weight->value, vocabulary_size_, hidden_size_) ||
        !valid_bias(bias->value, vocabulary_size_)) {
        set_error(error, "vocabulary projection parameters are missing or invalid");
        return false;
    }
    const std::size_t batch = hidden.shape()[0];
    const std::size_t sequence = hidden.shape()[1];
    if (sequence != 0 && batch > std::numeric_limits<std::size_t>::max() / sequence) {
        set_error(error, "vocabulary projection token count overflows size_t");
        return false;
    }
    if (!logits.reset({batch, sequence, vocabulary_size_}, TensorDataType::F32, TensorDevice::CPU, error)) {
        return false;
    }
    for (std::size_t token = 0; token < batch * sequence; ++token) {
        const std::size_t hidden_offset = token * hidden_size_;
        const std::size_t logits_offset = token * vocabulary_size_;
        for (std::size_t vocabulary = 0; vocabulary < vocabulary_size_; ++vocabulary) {
            double sum = static_cast<double>(bias->value.data()[vocabulary]);
            const std::size_t weight_offset = vocabulary * hidden_size_;
            for (std::size_t dimension = 0; dimension < hidden_size_; ++dimension) {
                sum += static_cast<double>(hidden.data()[hidden_offset + dimension]) *
                       static_cast<double>(weight->value.data()[weight_offset + dimension]);
            }
            const float value = static_cast<float>(sum);
            if (!std::isfinite(value)) {
                set_error(error, "vocabulary projection logits contain NaN or infinity");
                return false;
            }
            logits.data()[logits_offset + vocabulary] = value;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

bool VocabularyProjection::forward_last(const Tensor& hidden,
                                       const ParameterStore& parameters,
                                       Tensor& logits,
                                       std::string* error) const {
    if (vocabulary_size_ == 0 || hidden_size_ == 0) {
        set_error(error, "vocabulary projection is not initialized");
        return false;
    }
    if (!valid_hidden(hidden) || hidden.shape()[1] != 1) {
        set_error(error, "last-token projection requires a finite rank-3 tensor with sequence length one");
        return false;
    }
    const Parameter* weight = parameters.find(tied_embeddings_ ? "embedding.weight" : weight_name_);
    const Parameter* bias = parameters.find(bias_name_);
    if (weight == nullptr || bias == nullptr ||
        !valid_weight(weight->value, vocabulary_size_, hidden_size_) ||
        !valid_bias(bias->value, vocabulary_size_)) {
        set_error(error, "vocabulary projection parameters are missing or invalid");
        return false;
    }
    if (!logits.reset({hidden.shape()[0], vocabulary_size_}, TensorDataType::F32, TensorDevice::CPU, error)) {
        return false;
    }
    for (std::size_t batch = 0; batch < hidden.shape()[0]; ++batch) {
        const std::size_t hidden_offset = batch * hidden_size_;
        for (std::size_t vocabulary = 0; vocabulary < vocabulary_size_; ++vocabulary) {
            double sum = static_cast<double>(bias->value.data()[vocabulary]);
            const std::size_t weight_offset = vocabulary * hidden_size_;
            for (std::size_t dimension = 0; dimension < hidden_size_; ++dimension) {
                sum += static_cast<double>(hidden.data()[hidden_offset + dimension]) *
                       static_cast<double>(weight->value.data()[weight_offset + dimension]);
            }
            const float value = static_cast<float>(sum);
            if (!std::isfinite(value)) {
                set_error(error, "vocabulary projection logits contain NaN or infinity");
                return false;
            }
            logits.data()[batch * vocabulary_size_ + vocabulary] = value;
        }
    }
    if (error != nullptr) error->clear();
    return true;
}

std::size_t VocabularyProjection::vocabulary_size() const noexcept { return vocabulary_size_; }
std::size_t VocabularyProjection::hidden_size() const noexcept { return hidden_size_; }
bool VocabularyProjection::tied_embeddings() const noexcept { return tied_embeddings_; }
const std::string& VocabularyProjection::weight_name() const noexcept { return weight_name_; }
const std::string& VocabularyProjection::bias_name() const noexcept { return bias_name_; }

bool AutoregressiveLogits::register_parameters(const TransformerConfig& config,
                                              ParameterStore& parameters,
                                              std::string* error) {
    return projection_.register_parameters(config, parameters, error);
}

bool AutoregressiveLogits::forward_last(const Tensor& hidden,
                                       const ParameterStore& parameters,
                                       Tensor& logits,
                                       std::string* error) const {
    return projection_.forward_last(hidden, parameters, logits, error);
}

std::size_t AutoregressiveLogits::vocabulary_size() const noexcept { return projection_.vocabulary_size(); }
std::size_t AutoregressiveLogits::hidden_size() const noexcept { return projection_.hidden_size(); }
bool AutoregressiveLogits::tied_embeddings() const noexcept { return projection_.tied_embeddings(); }

} // namespace attention
