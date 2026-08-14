# Stage 2.4 Forward and Causal Loss Report

The first Stage 2.4 execution slice is complete through `attention::TransformerModel`. The model registers and composes token embeddings, sinusoidal positions, every configured `TransformerBlock`, final normalization, and vocabulary projection. It emits full `[batch, sequence, vocabulary]` logits.

The same model provides causal language-model loss with shifted next-token targets. The loss uses a stable log-sum-exp calculation, averages only over valid next-token predictions, and rejects malformed shapes, short sequences, out-of-range targets, nonfinite logits, and nonfinite results.

Verification completed:

| Gate | Result |
|---|---:|
| Focused TransformerModel tests | 3/3 passed |
| Full Release suite | 96/96 passed |
| Portable ASAN/UBSAN suite | 96/96 passed |
| Deterministic repeated forward and loss | Passed |
| Exact stable causal cross-entropy test | Passed |
| Invalid token, target, shape, and nonfinite-input tests | Passed |
| Forbidden token-pair complexity scan | Passed |

This completes the first two Stage 2.4 checklist items: complete forward pass and causal language-model loss. Backward propagation, gradient flow, numerical gradient checks, checkpoint equivalence, and malformed-configuration coverage remain pending and are not implied by this slice.

## Dedicated Deterministic-Forward Regression

A dedicated multi-batch test now runs the initialized model twice with identical token IDs and compares every full-logit scalar and causal-loss scalar. This supplements checkpoint reload equivalence by isolating in-process deterministic forward behavior under initialized nonzero parameters.
