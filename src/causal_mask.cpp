#include "attention/causal_mask.h"

namespace attention {

bool CausalMask::reset(std::size_t context_length,
                       std::string* error) noexcept {
    if (context_length == 0) {
        if (error != nullptr) *error = "context length must be positive";
        return false;
    }
    context_length_ = context_length;
    if (error != nullptr) error->clear();
    return true;
}

bool CausalMask::valid_sequence_length(std::size_t sequence_length) const noexcept {
    return context_length_ != 0 && sequence_length != 0 &&
           sequence_length <= context_length_;
}

bool CausalMask::allows(std::size_t query_position,
                        std::size_t key_position) const noexcept {
    return context_length_ != 0 && query_position < context_length_ &&
           key_position < context_length_ && key_position <= query_position;
}

std::size_t CausalMask::context_length() const noexcept {
    return context_length_;
}

std::size_t CausalMask::storage_bytes() const noexcept {
    return 0;
}

} // namespace attention
