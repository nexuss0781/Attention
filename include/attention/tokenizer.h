#ifndef ATTENTION_TOKENIZER_H
#define ATTENTION_TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace attention {

class ByteLevelTokenizer {
public:
    static constexpr std::size_t kByteVocabularySize = 256;
    static constexpr std::size_t kBeginningOfSequence = 256;
    static constexpr std::size_t kEndOfSequence = 257;
    static constexpr std::size_t kPadding = 258;
    static constexpr std::size_t kUnknown = 259;
    static constexpr std::size_t kVocabularySize = 260;

    [[nodiscard]] static constexpr std::string_view version() noexcept {
        return "attention.byte_utf8.v1";
    }

    bool encode(std::string_view text,
                std::vector<std::size_t>& tokens,
                bool add_beginning_of_sequence = true,
                bool add_end_of_sequence = true,
                std::string* error = nullptr) const;

    bool decode(const std::vector<std::size_t>& tokens,
                std::string& text,
                bool strip_special_tokens = true,
                std::string* error = nullptr) const;

    [[nodiscard]] static bool valid_utf8(std::string_view text,
                                         std::string* error = nullptr) noexcept;
    [[nodiscard]] static constexpr std::size_t vocabulary_size() noexcept {
        return kVocabularySize;
    }
};

} // namespace attention

#endif // ATTENTION_TOKENIZER_H
