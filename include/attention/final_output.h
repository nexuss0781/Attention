#ifndef ATTENTION_FINAL_OUTPUT_H
#define ATTENTION_FINAL_OUTPUT_H

#include "attention/normalization.h"

#include <cstddef>
#include <string>

namespace attention {

class FinalOutput {
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
    Normalization normalization_{"final_norm"};
};

} // namespace attention

#endif // ATTENTION_FINAL_OUTPUT_H
