#include "attention/final_output.h"

namespace attention {

bool FinalOutput::register_parameters(const TransformerConfig& config,
                                      ParameterStore& parameters,
                                      std::string* error) {
    return normalization_.register_parameters(config, parameters, error);
}

bool FinalOutput::forward(const Tensor& input,
                          const ParameterStore& parameters,
                          Tensor& output,
                          std::string* error) const {
    return normalization_.forward(input, parameters, output, error);
}

std::size_t FinalOutput::hidden_size() const noexcept {
    return normalization_.hidden_size();
}

float FinalOutput::epsilon() const noexcept {
    return normalization_.epsilon();
}

const std::string& FinalOutput::prefix() const noexcept {
    return normalization_.prefix();
}

} // namespace attention
