# Composable Transformer Block Report

The next ordered transformer milestone is complete through `attention::TransformerBlock`. The block composes the existing modules as a causal pre-normalization pipeline: first normalization, QKV projection, linear causal attention, residual addition, second normalization, feed-forward expansion/projection, and a final residual addition.

The block is independently parameterized by layer index. It registers two normalization namespaces, the existing `layers.<index>.attention.*` QKV namespace, and a new `layers.<index>.ffn.*` namespace. `FeedForward` now accepts an optional prefix while retaining the standalone default `ffn` prefix, so existing callers remain compatible.

Verification completed after the final causal-configuration hardening:

- Focused block tests passed before the final hardening: **3/3**.
- Full Release suite passed: **84/84** after the final causal-configuration hardening.
- Portable AddressSanitizer and UndefinedBehaviorSanitizer suite passed: **84/84** after the final causal-configuration hardening.
- Deterministic repeated-forward comparison passed.
- No dense attention matrix or token-pair storage was introduced.

The next ordered Todo item is Stage 2.1 completion: full inference-memory estimation and deterministic configuration serialization. Later Stage 2.4 work still includes the complete forward pass, causal language-model loss, backward propagation, gradient checks, determinism coverage, and checkpoint reload equivalence.
