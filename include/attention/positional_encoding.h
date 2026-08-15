#ifndef ATTENTION_POSITIONAL_ENCODING_H
#define ATTENTION_POSITIONAL_ENCODING_H

#include "attention/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace attention {

class SinusoidalPositionEncoding {
public:
    bool reset(std::size_t context_length,
               std::size_t hidden_size,
               std::string* error = nullptr) noexcept;

    bool apply(const Tensor& input,
               Tensor& output,
               std::string* error = nullptr) const;
    bool apply_at(const Tensor& input,
                  std::size_t position_offset,
                  Tensor& output,
                  std::string* error = nullptr) const;

    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] std::size_t hidden_size() const noexcept;

private:
    std::size_t context_length_ = 0;
    std::size_t hidden_size_ = 0;
    std::vector<float> table_;
};

} // namespace attention

#endif // ATTENTION_POSITIONAL_ENCODING_H
