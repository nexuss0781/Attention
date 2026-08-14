# Tokenizer Contract

`attention::ByteLevelTokenizer` is the first versioned tokenizer artifact for the Attention training and evaluation foundation. Its frozen version identifier is `attention.byte_utf8.v1`, and its vocabulary contains the 256 raw byte IDs followed by beginning-of-sequence, end-of-sequence, padding, and unknown IDs at 256–259.

The tokenizer preserves valid UTF-8 input byte-for-byte. It performs no whitespace, punctuation, capitalization, number, symbol, code, or language-specific normalization. This deliberately conservative v1 behavior avoids silent text changes while the project defines the multilingual and domain policy required for a production tokenizer. Malformed UTF-8 is rejected rather than replaced or mapped to the unknown token.

Encoding is deterministic and can add BOS and EOS boundaries independently. Decoding strips special tokens by default and validates the reconstructed byte sequence as canonical UTF-8. An explicit diagnostic mode renders special tokens as textual markers such as `<|bos|>` and `<|eos|>`. Unknown token IDs and malformed decoded byte sequences are rejected.

The artifact is suitable for deterministic small-scale training and evaluation plumbing, but it is not yet a subword tokenizer and no token-efficiency claim is made for English, Amharic, bilingual text, code, or structured messages. Those measurements and the final tokenizer choice remain later Stage 3 work.
