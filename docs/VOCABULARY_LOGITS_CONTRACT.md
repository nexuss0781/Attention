# Vocabulary Projection and Autoregressive Logits Contract

`attention::VocabularyProjection` maps hidden activations to vocabulary logits. It accepts F32 CPU row-major rank-3 tensors with shape `[batch, sequence, hidden]`.

For `TransformerConfig::tie_embeddings = true`, the projection reads the existing `embedding.weight` parameter with shape `[vocabulary, hidden]`; it does not register a duplicate output weight. For untied configurations it registers `lm_head.weight` with shape `[vocabulary, hidden]`. Both modes register `lm_head.bias` with shape `[vocabulary]`.

For each token, the logit for vocabulary row `v` is `hidden · weight[v] + bias[v]`. Full-sequence projection returns `[batch, sequence, vocabulary]` and uses memory proportional to the requested logits tensor. `attention::AutoregressiveLogits` additionally exposes a last-token path that consumes `[batch, 1, hidden]` and returns `[batch, vocabulary]`, avoiding storage for prior-token logits during generation.

Arithmetic is `O(batch × sequence × hidden × vocabulary)` for full projection and `O(batch × hidden × vocabulary)` for last-token projection. No query-key, token-pair, or sequence-pair matrix is created. All dimensions, parameter shapes, finite values, and output sums are checked. The module does not apply softmax; logits remain unnormalized for the later causal language-model loss and decoding stages.
