# Feed-Forward Layer Contract

The Attention feed-forward layer is a standalone F32 CPU module that maps a batch-major activation tensor of shape `[batch, sequence, hidden]` through an expansion, activation, and projection back to `hidden`.

For a configuration with hidden size `H` and feed-forward size `F`, the layer registers four stable parameters:

| Name | Shape | Role |
|---|---|---|
| `ffn.up.weight` | `[H, F]` | Input-to-expansion matrix |
| `ffn.up.bias` | `[F]` | Expansion bias |
| `ffn.down.weight` | `[F, H]` | Expansion-to-output matrix |
| `ffn.down.bias` | `[H]` | Output bias |

The forward calculation for each token is `u = activation(x · W_up + b_up)` followed by `y = u · W_down + b_down`. The declared configuration activation is used exactly: GELU uses the tanh approximation with the standard `sqrt(2/pi)` constants, and SiLU uses `x / (1 + exp(-x))` with finite-safe exponent handling.

The module validates configuration, parameter presence and shapes, rank, F32 CPU row-major metadata, exact hidden dimensions, and finite input and output values. It allocates only output and token-local expansion workspace. It does not create any sequence-by-sequence score, mask, or pairwise buffer. Arithmetic is `O(batch × sequence × hidden × feed_forward_size)` and extra working memory is `O(feed_forward_size)` per active token.

The first implementation is intentionally a plain expansion/projection block. Gated feed-forward variants remain a separate future contract because they require additional parameter naming and quality tests.
