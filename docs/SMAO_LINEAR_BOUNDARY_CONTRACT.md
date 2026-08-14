# SMAO and Linear-Attention Boundary Contract

**Status:** Stage 2.3 semantic-boundary decision

The existing SMAO Phase 1 kernel and the new linear feature-map attention path are related but not semantically identical. They must not be fused by silently reinterpreting one method’s outputs as the other method’s exact attention weights.

## SMAO Meaning

SMAO Phase 1 assembles a metric from the lower-triangular input, computes a conditioned metric and whitening operator, produces whitened query/key coordinates, and computes scalar decomposition factors for the Gaussian identity used by the numerical kernel. The scalar factors and whitened coordinates jointly belong to that decomposition.

## Safe Composition

The supported boundary adapter exposes the finite whitened query/key coordinates as `[1, sequence, hidden]` linear-attention inputs. This is a geometry-preserving preprocessing path: it lets the linear feature-map recurrence operate in the SMAO-whitened coordinate system without allocating token-pair scores.

The adapter explicitly reports that `query_scales` and `key_weights` were **not consumed** by the linear recurrence. They are not silently multiplied into the feature map, because doing so would claim an equivalence that is not established by the current mathematics.

## Incompatible Composition

The following combinations are rejected or remain unsupported until separately derived and tested:

| Combination | Status | Reason |
|---|---|---|
| Dense SMAO Gaussian pairwise scores | Rejected | Would create O(sequence²) pair work/storage |
| Treating SMAO scalar factors as linear-feature-map factors | Unsupported | Different decomposition semantics; no proven identity |
| Replacing the metric-aware whitening operator with an arbitrary feature-map transform | Unsupported | Changes the geometry contract |
| Feeding invalid, nonfinite, or failed SMAO outputs into the adapter | Rejected | Boundary cannot hide upstream failure |

## Complexity

The adapter copies two `sequence × hidden` coordinate matrices into batch-major rank-3 tensors, so its work and storage are `O(sequence × hidden)`. It stores no sequence-by-sequence matrix. The downstream `LinearCausalAttention` path remains `O(batch × sequence × hidden²)` arithmetic with dimension-bounded streaming state.

This boundary is intentionally conservative. It provides a tested composition point without claiming that linear feature-map attention reproduces SMAO’s exact Gaussian decomposition or dense softmax attention.
