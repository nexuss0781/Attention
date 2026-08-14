#ifndef ATTENTION_CAUSAL_MASK_H
#define ATTENTION_CAUSAL_MASK_H

#include <cstddef>
#include <string>

namespace attention {

class CausalMask {
public:
    bool reset(std::size_t context_length,
               std::string* error = nullptr) noexcept;

    [[nodiscard]] bool valid_sequence_length(std::size_t sequence_length) const noexcept;
    [[nodiscard]] bool allows(std::size_t query_position,
                              std::size_t key_position) const noexcept;
    [[nodiscard]] std::size_t context_length() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

private:
    std::size_t context_length_ = 0;
};

} // namespace attention

#endif // ATTENTION_CAUSAL_MASK_H
