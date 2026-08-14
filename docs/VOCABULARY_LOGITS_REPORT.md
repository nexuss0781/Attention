# Vocabulary Projection and Autoregressive Logits Report

The output head is now implemented through `attention::VocabularyProjection` and `attention::AutoregressiveLogits`.

Tied configurations reuse the existing `embedding.weight` tensor and register only `lm_head.bias`. Untied configurations register `lm_head.weight` and `lm_head.bias`. Both paths compute logits as a hidden-to-vocabulary dot product plus bias, without applying softmax.

The full path returns `[batch, sequence, vocabulary]` logits. The autoregressive last-token path consumes `[batch, 1, hidden]` and returns `[batch, vocabulary]` directly, avoiding an intermediate full-sequence logits tensor. Arithmetic is `O(batch × sequence × hidden × vocabulary)` for full projection and `O(batch × hidden × vocabulary)` for last-token generation. No query-key, token-pair, or sequence-pair matrix is allocated.

Verification completed:

- Native Release suite: **80/80 tests passed**.
- Portable AddressSanitizer and UndefinedBehaviorSanitizer suite: **80/80 tests passed**.
- No compiler diagnostics or sanitizer findings.
- Tests cover tied and untied parameter registration, exact logits, last-token output shape, missing tied embeddings, duplicate registration, and nonfinite input rejection.
- Live-source complexity scan found no forbidden `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` token-pair allocation pattern.

The next Todo item is composable transformer-block integration. Full causal language-model loss and training remain later Stage 2.4 work.
