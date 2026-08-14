# Embedding and Positional Representation Contract

**Status:** Stage 2.3 implementation slice

This contract defines the first composable input path for the transformer foundation. It deliberately stops before projections, attention scores, masking, feed-forward layers, normalization, residual blocks, and vocabulary logits.

## Token Embeddings

`attention::TokenEmbedding` owns no parameter storage itself. It registers or references the stable `embedding.weight` parameter in a `ParameterStore`, whose shape is `[vocabulary_size, hidden_size]`. A forward call accepts a flat batch of token IDs with an explicit `batch_size` and `sequence_length` and produces an F32 CPU tensor with shape `[batch_size, sequence_length, hidden_size]`.

Each token ID must be less than `vocabulary_size`. The input token vector must contain exactly `batch_size × sequence_length` IDs. Empty batches, empty sequences, and overflowing shape products are rejected. Output rows are exact copies of the corresponding embedding row; no normalization or positional signal is applied by this component.

## Positional Representation

`attention::SinusoidalPositionEncoding` adds a deterministic sinusoidal representation to an existing embedding tensor. It accepts and returns tensors with shape `[batch_size, sequence_length, hidden_size]`, requires a sequence length no greater than the configured context length, and does not allocate or register trainable parameters.

For channel pair `2i` and `2i+1`, position `p` uses:

```text
angle(p, i) = p / 10000^(2i / hidden_size)
output[..., 2i]     += sin(angle(p, i))
output[..., 2i + 1] += cos(angle(p, i))
```

For an odd hidden size, the final unpaired channel receives the sine component only. The representation is batch-independent, deterministic, finite for supported shapes, and applied without changing the input tensor’s shape.

## Composition Boundary

The intended first input path is:

```text
Token IDs → TokenEmbedding → SinusoidalPositionEncoding → future projection/attention block
```

The embedding parameter name is checkpoint-compatible and stable. Later work may add learned positions, rotary positions, or another representation, but such changes must introduce a versioned contract rather than silently changing this implementation’s semantics.
