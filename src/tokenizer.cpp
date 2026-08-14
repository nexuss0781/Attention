#include "attention/tokenizer.h"

#include <limits>

namespace attention {
namespace {

void set_error(std::string* error, const char* message) noexcept {
    if (error != nullptr) *error = message;
}

bool continuation(unsigned char value) noexcept {
    return (value & 0xc0u) == 0x80u;
}

} // namespace

bool ByteLevelTokenizer::valid_utf8(std::string_view text, std::string* error) noexcept {
    for (std::size_t index = 0; index < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        std::uint32_t code_point = 0;
        if (first <= 0x7fu) {
            length = 1;
            code_point = first;
        } else if (first >= 0xc2u && first <= 0xdfu) {
            length = 2;
            code_point = first & 0x1fu;
        } else if (first >= 0xe0u && first <= 0xefu) {
            length = 3;
            code_point = first & 0x0fu;
        } else if (first >= 0xf0u && first <= 0xf4u) {
            length = 4;
            code_point = first & 0x07u;
        } else {
            set_error(error, "tokenizer input contains an invalid UTF-8 lead byte");
            return false;
        }
        if (index + length > text.size()) {
            set_error(error, "tokenizer input contains truncated UTF-8");
            return false;
        }
        for (std::size_t offset = 1; offset < length; ++offset) {
            const unsigned char value = static_cast<unsigned char>(text[index + offset]);
            if (!continuation(value)) {
                set_error(error, "tokenizer input contains an invalid UTF-8 continuation byte");
                return false;
            }
            code_point = (code_point << 6u) | (value & 0x3fu);
        }
        if ((length == 2 && code_point < 0x80u) ||
            (length == 3 && code_point < 0x800u) ||
            (length == 4 && code_point < 0x10000u) ||
            code_point > 0x10ffffu ||
            (code_point >= 0xd800u && code_point <= 0xdfffu)) {
            set_error(error, "tokenizer input contains a noncanonical UTF-8 code point");
            return false;
        }
        index += length;
    }
    if (error != nullptr) error->clear();
    return true;
}

bool ByteLevelTokenizer::encode(std::string_view text,
                                std::vector<std::size_t>& tokens,
                                bool add_beginning_of_sequence,
                                bool add_end_of_sequence,
                                std::string* error) const {
    if (!valid_utf8(text, error)) return false;
    if (text.size() > std::numeric_limits<std::size_t>::max() -
                          static_cast<std::size_t>(add_beginning_of_sequence) -
                          static_cast<std::size_t>(add_end_of_sequence)) {
        set_error(error, "tokenizer output size overflows size_t");
        return false;
    }
    tokens.clear();
    tokens.reserve(text.size() + static_cast<std::size_t>(add_beginning_of_sequence) +
                   static_cast<std::size_t>(add_end_of_sequence));
    if (add_beginning_of_sequence) tokens.push_back(kBeginningOfSequence);
    for (const unsigned char byte : text) tokens.push_back(static_cast<std::size_t>(byte));
    if (add_end_of_sequence) tokens.push_back(kEndOfSequence);
    if (error != nullptr) error->clear();
    return true;
}

bool ByteLevelTokenizer::decode(const std::vector<std::size_t>& tokens,
                                std::string& text,
                                bool strip_special_tokens,
                                std::string* error) const {
    text.clear();
    text.reserve(tokens.size());
    for (const std::size_t token : tokens) {
        if (token < kByteVocabularySize) {
            text.push_back(static_cast<char>(static_cast<unsigned char>(token)));
        } else if (token == kBeginningOfSequence) {
            if (!strip_special_tokens) text.append("<|bos|>");
        } else if (token == kEndOfSequence) {
            if (!strip_special_tokens) text.append("<|eos|>");
        } else if (token == kPadding) {
            if (!strip_special_tokens) text.append("<|pad|>");
        } else if (token == kUnknown) {
            if (!strip_special_tokens) text.append("<|unk|>");
        } else {
            set_error(error, "tokenizer input contains an unknown token ID");
            text.clear();
            return false;
        }
    }
    if (!valid_utf8(text, error)) {
        text.clear();
        return false;
    }
    if (error != nullptr) error->clear();
    return true;
}

} // namespace attention
