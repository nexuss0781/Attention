#ifndef ATTENTION_FEED_FORWARD_H
#define ATTENTION_FEED_FORWARD_H

#include "attention/parameter_store.h"
#include "attention/tensor.h"
#include "attention/transformer_config.h"

#include <cstddef>
#include <string>

namespace attention {

class FeedForward {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const Tensor& input,
                 const ParameterStore& parameters,
                 Tensor& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] std::size_t feed_forward_size() const noexcept;
    [[nodiscard]] Activation activation() const noexcept;
    [[nodiscard]] const std::string& prefix() const noexcept;

private:
    std::size_t hidden_size_ = 0;
    std::size_t feed_forward_size_ = 0;
    Activation activation_ = Activation::GELU;
    std::string prefix_ = "ffn";
};

} // namespace attention

#endif // ATTENTION_FEED_FORWARD_H
