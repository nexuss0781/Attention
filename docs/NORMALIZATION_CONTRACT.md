# Normalization Contract

The first transformer normalization module is per-token LayerNorm over the final hidden dimension of a batch-major F32 CPU tensor with shape `[batch, sequence, hidden]`.

The module registers two stable parameters:

| Name | Shape | Role |
|---|---|---|
| `norm.weight` | `[hidden]` | Learned scale, initialized by the parameter store |
| `norm.bias` | `[hidden]` | Learned offset, initialized by the parameter store |

For each token vector `x`, the output is `y = weight * (x - mean(x)) / sqrt(variance(x) + epsilon) + bias`, where variance is the population variance over hidden units and `epsilon = 1e-5`. Mean and variance use F64 accumulation before the F32 output is written.

The module validates rank, F32 CPU row-major metadata, exact hidden size, parameter shape and finiteness, and finite output. It uses a token-local scalar workspace only. Its arithmetic is `O(batch × sequence × hidden)` and it allocates no sequence-pair or token-pair state. RMSNorm and alternative normalization placement remain separate future contracts.
