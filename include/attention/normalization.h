#ifndef ATTENTION_NORMALIZATION_H
#define ATTENTION_NORMALIZATION_H

#include "attention/parameter_store.h"
#include "attention/tensor.h"
#include "attention/transformer_config.h"

#include <cstddef>
#include <string>

namespace attention {

class Normalization {
public:
    bool register_parameters(const TransformerConfig& config,
                             ParameterStore& parameters,
                             std::string* error = nullptr);

    bool forward(const Tensor& input,
                 const ParameterStore& parameters,
                 Tensor& output,
                 std::string* error = nullptr) const;

    [[nodiscard]] std::size_t hidden_size() const noexcept;
    [[nodiscard]] float epsilon() const noexcept;
    [[nodiscard]] const std::string& prefix() const noexcept;

private:
    std::size_t hidden_size_ = 0;
    std::string prefix_ = "norm";
};

} // namespace attention

#endif // ATTENTION_NORMALIZATION_H
