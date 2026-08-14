# Projection and Causal-Mask Contract

**Status:** Next linear-scaling Stage 2.3 slice

This contract adds query, key, and value projections without implementing a quadratic attention-score matrix. It also defines a streaming causal-mask predicate rather than storing a `[sequence, sequence]` mask.

## QKV Projection

`attention::QKVProjection` registers six stable parameters for one configured transformer layer:

```text
layers.<index>.attention.q_proj.weight
layers.<index>.attention.q_proj.bias
layers.<index>.attention.k_proj.weight
layers.<index>.attention.k_proj.bias
layers.<index>.attention.v_proj.weight
layers.<index>.attention.v_proj.bias
```

Each weight has shape `[hidden_size, hidden_size]` and each bias has shape `[hidden_size]`. The projection input and each output have shape `[batch_size, sequence_length, hidden_size]`, stored as contiguous row-major F32 CPU tensors. For each token row, the implementation computes three independent matrix-vector products.

The projection work is `O(batch_size × sequence_length × hidden_size²)` and its temporary/output storage is `O(batch_size × sequence_length × hidden_size)`. It does not compare one token with another and does not allocate any token-pair buffer.

## Causal Mask

`attention::CausalMask` stores only the configured context length. `allows(query_position, key_position)` returns true exactly when `key_position <= query_position`, and false otherwise. A sequence length is valid when it is positive and no greater than the context length.

The mask has zero sequence-dependent storage. Consumers must apply it while streaming over keys for one query position or while using a future linear-attention accumulator; they must not expand it into a dense `[sequence_length, sequence_length]` tensor.

## Explicit Non-Goal

This slice does not implement softmax attention or a dense score matrix. The later aggregation design must preserve the project invariant: the core Attention path remains linear in token count and may use only dimension-dependent state and per-token outputs. Any operation that requires materializing all query-key pairs must be rejected or redesigned before implementation.
