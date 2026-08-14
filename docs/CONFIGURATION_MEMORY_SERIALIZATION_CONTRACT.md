# Configuration Memory and Serialization Contract

`attention::TransformerConfig` now exposes `estimated_inference_memory_bytes(batch_size, resident_sequence_length)`. The estimate includes parameter storage, resident hidden-width block workspaces, one feed-forward expansion workspace, token IDs, direct last-token vocabulary logits, and the dimension-bounded double-precision linear-attention state. When `resident_sequence_length` is zero, the configured context length is used. A resident length greater than the configured context, a zero batch, an invalid configuration, or any arithmetic overflow returns zero.

The estimate intentionally distinguishes **logical context length** from **resident sequence length**. A streamed 100M- or 1B-token logical context does not force a context-sized activation allocation; the estimate can be evaluated for the actual resident chunk while the recurrent attention state remains bounded by batch and hidden dimensions.

The serialization format is a strict versioned text contract:

```text
attention.transformer_config.v1
vocabulary_size=<uint>
context_length=<uint>
layer_count=<uint>
hidden_size=<uint>
attention_head_count=<uint>
feed_forward_size=<uint>
activation=<0|1>
precision=<0>
tie_embeddings=<0|1>
causal=<0|1>
dropout_probability=<finite float>
```

Fields are emitted in the fixed order above with no locale dependence. Serialization validates the configuration first. Deserialization requires the exact header, exact field order, valid scalar representations, no missing fields, and no trailing data; the reconstructed configuration is validated before it is returned. This makes serialized output deterministic and suitable for checkpoint metadata and reproducibility checks.
