#ifndef ATTENTION_CHECKPOINT_H
#define ATTENTION_CHECKPOINT_H

#include "attention/parameter_store.h"
#include "attention/tokenizer.h"
#include "attention/transformer_model.h"

#include <string>
#include <string_view>

namespace attention {

struct TokenizerMetadata {
    std::string version;
    std::size_t vocabulary_size = 0;
    std::size_t beginning_of_sequence = 0;
    std::size_t end_of_sequence = 0;
    std::size_t padding = 0;
    std::size_t unknown = 0;

    [[nodiscard]] static TokenizerMetadata byte_level_v1();
    [[nodiscard]] bool validate(std::string* error = nullptr) const;
    [[nodiscard]] bool operator==(const TokenizerMetadata& other) const noexcept;
};

class TransformerCheckpoint {
public:
    static bool serialize(const TransformerConfig& config,
                          const ParameterStore& parameters,
                          std::string& output,
                          std::string* error = nullptr,
                          const TokenizerMetadata& tokenizer = TokenizerMetadata::byte_level_v1());

    static bool load(std::string_view input,
                     TransformerModel& model,
                     ParameterStore& parameters,
                     std::string* error = nullptr,
                     const TokenizerMetadata& expected_tokenizer = TokenizerMetadata::byte_level_v1());
};

} // namespace attention

#endif // ATTENTION_CHECKPOINT_H
