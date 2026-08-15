# Stage 0 Causal-Basics Evaluation Result

## Scope

This report evaluates the saved `causal_basics` checkpoint with four fixed English continuation prompts. It is a deterministic greedy-decoding probe, not a broad language-quality benchmark. The model configuration is intentionally tiny: vocabulary size 260, context length 4, one layer, hidden size 2, and feed-forward size 4.

## Measured output

| Prompt | Window loss | Greedy continuation token IDs |
|---|---:|---|
| `The sky is` | `5.33349133` | `34 34 34 34 34 34 34 34` |
| `Once upon a time` | `5.42229033` | `34 34 34 34 34 34 34 34` |
| `A cat sat on the` | `5.39163923` | `34 34 34 34 34 34 34 34` |
| `The capital of France is` | `5.33349133` | `34 34 34 34 34 34 34 34` |

Repeated evaluation produced byte-for-byte identical reports. The Release suite remains `125/125` passing, and the live-source complexity scan found no targeted quadratic token-pair allocation pattern.

## Interpretation

The checkpoint demonstrates a measurable training signal and valid checkpoint/reload mechanics, but it does **not** demonstrate useful English generation. The repeated token `34` is a degenerate greedy policy for this tiny model and tiny 32-token training budget. This is expected at the current diagnostic scale and should not be described as language mastery.

The correct Stage 0 decision is therefore:

> **Pipeline competency: pass. English language competency: not yet demonstrated.**

The next session must use a larger bounded English chunk and a fixed held-out continuation set before the model can be promoted to the local-coherence competency. The current correctness-first finite-difference backward implementation makes the token budget computationally expensive; scaling should remain bounded until an analytical backward kernel is available.
