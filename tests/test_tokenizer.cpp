#include "attention/tokenizer.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace attention {
namespace {

TEST(ByteLevelTokenizerTest, HasVersionedStableVocabularyAndDeterministicEncoding) {
    ByteLevelTokenizer tokenizer;
    const std::string text = "Attention ሰላም — linear memory";
    std::vector<std::size_t> first;
    std::vector<std::size_t> second;
    std::string error;
    ASSERT_TRUE(tokenizer.encode(text, first, true, true, &error)) << error;
    ASSERT_TRUE(tokenizer.encode(text, second, true, true, &error)) << error;
    EXPECT_EQ(ByteLevelTokenizer::version(), "attention.byte_utf8.v1");
    EXPECT_EQ(ByteLevelTokenizer::vocabulary_size(), 260u);
    EXPECT_EQ(first, second);
    ASSERT_EQ(first.front(), ByteLevelTokenizer::kBeginningOfSequence);
    ASSERT_EQ(first.back(), ByteLevelTokenizer::kEndOfSequence);
}

TEST(ByteLevelTokenizerTest, RoundTripsUnicodeCodeAndStructuredText) {
    ByteLevelTokenizer tokenizer;
    const std::vector<std::string> samples{
        "plain English text",
        "አማርኛ English",
        "int main() { return 0; }\n",
        "{\"role\":\"tool\",\"content\":\"ok\"}",
        "emoji: 🚀 and combining: e\xCC\x81"};
    for (const std::string& sample : samples) {
        std::vector<std::size_t> tokens;
        std::string decoded;
        std::string error;
        ASSERT_TRUE(tokenizer.encode(sample, tokens, true, true, &error)) << sample << ": " << error;
        ASSERT_TRUE(tokenizer.decode(tokens, decoded, true, &error)) << sample << ": " << error;
        EXPECT_EQ(decoded, sample);
    }
}

TEST(ByteLevelTokenizerTest, SupportsExplicitBoundaryPolicyAndSpecialRendering) {
    ByteLevelTokenizer tokenizer;
    std::vector<std::size_t> tokens;
    std::string error;
    ASSERT_TRUE(tokenizer.encode("abc", tokens, false, false, &error)) << error;
    EXPECT_EQ(tokens, (std::vector<std::size_t>{'a', 'b', 'c'}));
    std::string decoded;
    ASSERT_TRUE(tokenizer.decode({ByteLevelTokenizer::kBeginningOfSequence, 'a',
                                  ByteLevelTokenizer::kEndOfSequence}, decoded, true, &error)) << error;
    EXPECT_EQ(decoded, "a");
    ASSERT_TRUE(tokenizer.decode({ByteLevelTokenizer::kBeginningOfSequence, 'a',
                                  ByteLevelTokenizer::kEndOfSequence}, decoded, false, &error)) << error;
    EXPECT_EQ(decoded, "<|bos|>a<|eos|>");
}

TEST(ByteLevelTokenizerTest, RejectsMalformedUnicodeAndUnknownTokenIDs) {
    ByteLevelTokenizer tokenizer;
    std::vector<std::size_t> tokens;
    std::string error;
    EXPECT_FALSE(tokenizer.encode(std::string("bad\xC3\x28", 5), tokens, true, true, &error));
    EXPECT_NE(error.find("UTF-8"), std::string::npos);
    std::string decoded;
    EXPECT_FALSE(tokenizer.decode({260}, decoded, true, &error));
    EXPECT_NE(error.find("unknown token"), std::string::npos);
    EXPECT_FALSE(tokenizer.decode({0xff}, decoded, true, &error));
    EXPECT_NE(error.find("UTF-8"), std::string::npos);
}

} // namespace
} // namespace attention
