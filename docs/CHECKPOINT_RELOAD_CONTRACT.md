# Checkpoint Reload Contract

`attention::TransformerCheckpoint` serializes a validated `TransformerConfig` and the finite values of every registered `ParameterStore` parameter into the versioned `attention.checkpoint.v1` text format. The configuration is framed by an explicit byte count. Parameters are emitted in strict lexicographic name order, with explicit name length, rank, shape, value count, and one deterministic round-trippable float per line.

Reload is allowed only into a fresh `TransformerModel` and empty `ParameterStore`. The loader parses and validates the embedded configuration, registers the model from that configuration, then requires every serialized parameter name, rank, shape, and scalar count to match the registered model exactly. Values are copied only after those checks and the resulting store must remain finite.

Malformed magic, configuration, lengths, ordering, names, ranks, dimensions, value counts, scalar values, trailing data, duplicate parameters, model reuse, and nonempty destination stores are rejected. A successful reload must produce bitwise-equivalent parameter values and identical forward logits for the same token stream.

The checkpoint contains only model configuration and parameter values. It does not contain token sequences, attention score matrices, or context-sized token-pair state.
