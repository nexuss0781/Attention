# Composable Transformer Block Contract

`attention::TransformerBlock` is a layer-scoped, causal pre-normalization composition of the existing modules. For layer `i`, the forward path is:

```text
x1 = LayerNorm(prefix = layers.i.norm1, input)
(q, k, v) = QKVProjection(layer = i, x1)
a = LinearCausalAttention(q, k, v)
y = input + a
x2 = LayerNorm(prefix = layers.i.norm2, y)
f = FeedForward(prefix = layers.i.ffn, x2)
output = y + f
```

The block owns no duplicate attention mathematics. It delegates normalization, QKV projection, linear causal aggregation, feed-forward expansion/projection, and residual addition to their independently testable modules. Registration is deterministic and uses the stable namespaces `layers.i.norm1.*`, `layers.i.attention.*`, `layers.i.norm2.*`, and `layers.i.ffn.*`.

The block accepts only causal configurations and finite F32 CPU row-major rank-3 tensors shaped `[batch, sequence, hidden]`, with sequence length no greater than the configured context. It produces a same-shape output and propagates failures from each submodule.

The composition has linear token dependence. Projection and feed-forward work scale with the resident token count and feature dimensions, while causal aggregation uses the existing feature-map state. No dense softmax matrix, token-pair buffer, or sequence-length-squared allocation is introduced by the block.

The focused tests verify stable names, exact causal composition using constant values, deterministic repeated forward execution, context overflow rejection, nonfinite input rejection, duplicate registration rejection, noncausal configuration rejection, and layer-index validation.
