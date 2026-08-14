# Stage 2.3 Projection and Causal-Mask Report

**Verification date:** 2026-08-14

**Branch:** `master`

## Scope Completed

This slice implements the query, key, and value projection boundary and a streaming causal-mask predicate. It does not implement softmax attention, dense pairwise scores, value aggregation, feed-forward layers, normalization, logits, loss, or backward propagation.

`attention::QKVProjection` registers stable per-layer weights and biases and maps `[batch, sequence, hidden]` inputs to three tensors of the same shape. `attention::CausalMask` stores only the configured context length and answers whether a key position is visible to a query position.

## Complexity Invariant

For fixed hidden size `h`, batch size `b`, and sequence length `t`, projection work is `O(b × t × h²)` and projection outputs use `O(b × t × h)` storage. The causal mask uses **zero sequence-dependent storage** and constant-time position checks. Neither component constructs, stores, or iterates over all `t²` query-key pairs.

The later attention aggregation must preserve this invariant. A dense `[t, t]` score or mask tensor is explicitly outside this contract and must not be introduced as an implementation shortcut.

## Stable Parameters

| Parameter | Shape |
|---|---:|
| `layers.<index>.attention.q_proj.weight` | `[hidden, hidden]` |
| `layers.<index>.attention.q_proj.bias` | `[hidden]` |
| `layers.<index>.attention.k_proj.weight` | `[hidden, hidden]` |
| `layers.<index>.attention.k_proj.bias` | `[hidden]` |
| `layers.<index>.attention.v_proj.weight` | `[hidden, hidden]` |
| `layers.<index>.attention.v_proj.bias` | `[hidden]` |

## Verification

The Release build with native SIMD and OpenMP passed **53/53 tests**. The tests verify exact deterministic projection outputs, stable parameter registration, shape mismatch rejection, NaN rejection, causal ordering, context bounds, and zero sequence-dependent mask storage.

The QKV output path is intentionally a per-token matrix-vector path. It has no token-pair allocation, no nested query-token/key-token loop, and no dense causal-mask representation. The existing project-wide complexity audit remains the governing constraint for all subsequent attention work.

## Next Boundary

The next component must implement a linear attention aggregation strategy using streaming or dimension-dependent state. It must be designed and benchmarked before implementation; standard dense softmax attention is not an acceptable default because it violates the project’s core linear-token constraint.
