# Transformer Forward and Causal Loss Contract

`attention::TransformerModel` composes the implemented architecture in the following order:

```text
TokenEmbedding
→ SinusoidalPositionEncoding
→ TransformerBlock[0..layer_count-1]
→ FinalOutput normalization
→ VocabularyProjection
→ full logits [batch, sequence, vocabulary]
```

The input token vector must contain exactly `batch × sequence` IDs, the sequence must be within the configured context, and every ID must be within the configured vocabulary. The output is full per-position vocabulary logits and remains unnormalized; softmax is applied only inside the loss calculation.

`causal_loss` uses the next token at position `t+1` as the target for logits at position `t`, excluding the final position because it has no next-token target. `causal_cross_entropy` computes each token loss with a numerically stable log-sum-exp reduction using the row maximum, then averages over `batch × (sequence - 1)` predictions. It rejects sequences shorter than two positions, malformed target lengths, out-of-range targets, nonfinite logits, and nonfinite loss results.

Forward and loss computation scale with resident token count, hidden dimensions, feed-forward dimensions, and vocabulary size. They do not construct dense query-key score matrices, token-pair buffers, or any sequence-length-squared structure. The full logits tensor is intentionally `[batch, sequence, vocabulary]`; direct last-token generation remains available through the existing output-head API for memory-sensitive autoregressive serving.
