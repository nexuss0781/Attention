# Linear Causal Attention Aggregation Contract

**Status:** Next Stage 2.3 implementation slice

This contract defines the first attention aggregation path that satisfies Attention’s core complexity requirement. It is a **linear causal feature-map attention** path, not an exact dense softmax-attention implementation.

## Feature Map

For each hidden channel, use the positive finite feature map:

```text
φ(x) = exp(clamp(x, -20, 20))
```

The bounded exponent keeps the feature map finite for valid F32 inputs and gives a strictly positive denominator.

## Streaming Recurrence

For each batch independently, process positions from left to right. At position `t`, first update the causal state with the current key and value:

```text
S_t[a,b] = S_{t-1}[a,b] + φ(K_t[a]) V_t[b]
z_t[a]    = z_{t-1}[a] + φ(K_t[a])
```

Then emit:

```text
Y_t[b] = (Σ_a φ(Q_t[a]) S_t[a,b]) /
         max(Σ_a φ(Q_t[a]) z_t[a], ε)
```

The current token is visible to itself, and only earlier or current tokens contribute to the output. The causal mask is enforced by the sequential update order and validated through `CausalMask`; no dense mask is created.

## Complexity and Storage Invariants

For batch size `B`, sequence length `T`, and hidden size `H`, the recurrence requires `O(B × T × H²)` arithmetic, `O(B × H²)` streaming state, and `O(B × T × H)` output storage. It never materializes `QKᵀ`, a dense score matrix, a dense causal mask, or any `T × T` buffer.

The method is intentionally explicit about its semantics. It is not claimed to equal the repository’s earlier exact Gaussian decomposition or dense softmax attention. The SMAO boundary remains separate until a later contract proves that its metric-aware geometry can be used without changing the linear recurrence’s meaning.

## Numerical Requirements

Inputs must be valid finite F32 CPU row-major rank-3 tensors with matching `[B,T,H]` shapes. The exponent is clipped to `[-20,20]`; accumulation uses F64 state before conversion to F32 output. The denominator must be finite and strictly positive after the configured epsilon floor. Outputs must be finite.
