# Stage 2.3 Linear Attention Aggregation Report

**Verification date:** 2026-08-14

**Branch:** `master`

## Scope Completed

This slice implements `attention::LinearCausalAttention`, a causal positive-feature-map attention path with streaming state. It is intentionally **not** an exact dense softmax-attention implementation. The current method uses `φ(x) = exp(clamp(x, -20, 20))`, updates a per-batch key-feature/value accumulator and normalizer from left to right, and emits each output from the current causal prefix.

The current token is visible to itself. Earlier tokens remain in the accumulator. No query-key score matrix, dense causal mask, or token-pair buffer is allocated.

## Complexity and Storage

For batch size `B`, sequence length `T`, and hidden size `H`:

| Quantity | Bound |
|---|---:|
| Arithmetic | `O(B × T × H²)` |
| Output storage | `O(B × T × H)` |
| Streaming state | `O(B × H²)` |
| Dense token-pair storage | Prohibited; none allocated |

The public `state_bytes(batch_size)` method reports only dimension- and batch-dependent state. It is independent of sequence length; for the benchmark configuration `B=1`, `H=8`, it reports 576 bytes for all tested sequence lengths.

## Numerical Behavior

The implementation validates matching F32 CPU row-major rank-3 tensors, context bounds, hidden size, and finite inputs. Feature exponents are clipped to `[-20,20]`. Accumulation uses F64 state and the denominator is protected by a positive epsilon floor. Output finiteness is checked before success.

The method’s semantics are explicitly separated from the existing SMAO Gaussian decomposition and from dense softmax attention. The SMAO integration boundary remains pending until a compatible metric-aware linear recurrence is defined and tested.

## Verification

The native Release build passed **56/56 tests**. The portable AddressSanitizer and UndefinedBehaviorSanitizer build also passed **56/56 tests**, with no compiler diagnostics or sanitizer findings.

The dedicated benchmark used `B=1`, `H=8`, three warmed samples, and sequence lengths of 10,000, 100,000, and 1,000,000:

| Sequence length | p95 time | p95 throughput | Reported state |
|---:|---:|---:|---:|
| 10,000 | 3.375 ms | 2.963 million tokens/sec | 576 bytes |
| 100,000 | 35.196 ms | 2.841 million tokens/sec | 576 bytes |
| 1,000,000 | 358.242 ms | 2.791 million tokens/sec | 576 bytes |

The approximately tenfold time increase for each tenfold sequence increase, together with constant reported state, is consistent with linear token scaling for fixed hidden size. A live-source scan found no `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` allocation pattern.

## Remaining Stage 2.3 Work

The next checklist item is the Attention/SMAO semantic boundary. It must be addressed without weakening the linear invariant. Feed-forward layers, normalization, residual connections, final output processing, vocabulary projection, autoregressive logits, complete block composition, configuration serialization, inference-memory estimation, and Stage 2.4 training execution remain pending.
