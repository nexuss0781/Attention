#ifndef ATTENTION_TOKEN_EMBEDDING_H
#define ATTENTION_TOKEN_EMBEDDING_H

#include "attention/parameter_store.h"
#include "attention/transformer_config.h"
#include "attention/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

class TokenEmbedding {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const std::vector<std::size_t>& token_ids,
                 std::size_t batch_size,
                 std::size_t sequence_length,
                 const ParameterStore& parameters,
                 Tensor& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] const std::string& parameter_name() const noexcept;

private:
    std::size_t vocabulary_size_ = 0;
    std::size_t hidden_size_ = 0;
    std::string parameter_name_ = "embedding.weight";
};

} // namespace attention

#endif // ATTENTION_TOKEN_EMBEDDING_H
