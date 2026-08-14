#ifndef ATTENTION_TRANSFORMER_BLOCK_H
#define ATTENTION_TRANSFORMER_BLOCK_H

#include "attention/feed_forward.h"
#include "attention/linear_attention.h"
#include "attention/normalization.h"
#include "attention/qkv_projection.h"
#include "attention/residual_connection.h"
#include "attention/transformer_config.h"

#include <cstddef>
#include <string>

namespace attention {

class TransformerBlock {
public:
    explicit TransformerBlock(std::size_t layer_index = 0);

    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const Tensor& input,
                 const ParameterStore& parameters,
                 Tensor& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    std::size_t layer_index_ = 0;
    std::size_t hidden_size_ = 0;
    std::size_t context_length_ = 0;
    bool initialized_ = false;
    Normalization attention_normalization_;
    QKVProjection qkv_projection_;
    LinearCausalAttention attention_;
    Normalization feed_forward_normalization_;
    FeedForward feed_forward_;
    ResidualConnection residual_;
};

} // namespace attention

#endif // ATTENTION_TRANSFORMER_BLOCK_H
