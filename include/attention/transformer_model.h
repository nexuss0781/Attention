#ifndef ATTENTION_TRANSFORMER_MODEL_H
#define ATTENTION_TRANSFORMER_MODEL_H

#include "attention/final_output.h"
#include "attention/positional_encoding.h"
#include "attention/token_embedding.h"
#include "attention/transformer_block.h"
#include "attention/vocabulary_projection.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

class TransformerModel {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const std::vector<std::size_t>& token_ids,
                 std::size_t batch_size,
                 std::size_t sequence_length,
                 const ParameterStore& parameters,
                 Tensor& logits,
                 std::string* error = nullptr) const;

    bool causal_loss(const std::vector<std::size_t>& token_ids,
                     std::size_t batch_size,
                     std::size_t sequence_length,
                     const ParameterStore& parameters,
                     float& loss,
                     std::string* error = nullptr) const;

    // Computes central finite-difference gradients for every registered parameter.
    // This correctness-first fallback is intentionally explicit and is not a
    // replacement for the later analytical backward kernel.
    bool backward(const std::vector<std::size_t>& token_ids,
                  std::size_t batch_size,
                  std::size_t sequence_length,
                  ParameterStore& parameters,
                  float difference_step = 1e-3f,
                  std::string* error = nullptr) const;

    static bool causal_cross_entropy(const Tensor& logits,
                                     const std::vector<std::size_t>& target_token_ids,
                                     float& loss,
                                     std::string* error = nullptr);

    [[nodiscard]] const TransformerConfig& config() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    TransformerConfig config_;
    bool initialized_ = false;
    TokenEmbedding embedding_;
    SinusoidalPositionEncoding positional_encoding_;
    std::vector<TransformerBlock> blocks_;
    FinalOutput final_output_;
    VocabularyProjection output_projection_;
};

} // namespace attention

#endif // ATTENTION_TRANSFORMER_MODEL_H
