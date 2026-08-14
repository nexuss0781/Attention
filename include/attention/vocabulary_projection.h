#ifndef ATTENTION_VOCABULARY_PROJECTION_H
#define ATTENTION_VOCABULARY_PROJECTION_H

#include "attention/parameter_store.h"
#include "attention/tensor.h"
#include "attention/transformer_config.h"

#include <cstddef>
#include <string>

namespace attention {

class VocabularyProjection {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const Tensor& hidden,
                 const ParameterStore& parameters,
                 Tensor& logits,
                 std::string* error = nullptr) const;

    bool forward_last(const Tensor& hidden,
                      const ParameterStore& parameters,
                      Tensor& logits,
                      std::string* error = nullptr) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] bool tied_embeddings() const noexcept;
    [[nodiscard]] const std::string& weight_name() const noexcept;
    [[nodiscard]] const std::string& bias_name() const noexcept;

private:
    std::size_t vocabulary_size_ = 0;
    std::size_t hidden_size_ = 0;
    bool tied_embeddings_ = true;
    std::string weight_name_ = "lm_head.weight";
    std::string bias_name_ = "lm_head.bias";
};

class AutoregressiveLogits {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward_last(const Tensor& hidden,
                      const ParameterStore& parameters,
                      Tensor& logits,
                      std::string* error = nullptr) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] bool tied_embeddings() const noexcept;

private:
    VocabularyProjection projection_;
};

} // namespace attention

#endif // ATTENTION_VOCABULARY_PROJECTION_H
