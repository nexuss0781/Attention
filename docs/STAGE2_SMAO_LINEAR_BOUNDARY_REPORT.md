# Stage 2.3 SMAO–Linear Attention Boundary Report

**Verification date:** 2026-08-14

**Branch:** `master`

## Boundary Decision

The existing SMAO Phase 1 kernel and the linear feature-map attention path are **composable only through finite whitened query/key coordinates**. The new `attention::SMAOLinearBoundary` adapter copies those coordinates into batch-major `[1, sequence, hidden]` tensors.

The adapter deliberately does **not** consume SMAO `query_scales` or `key_weights`. Those scalars belong to SMAO’s Gaussian decomposition identity, while the linear attention path uses a separate positive feature map. Treating them as interchangeable would claim an unproven mathematical equivalence, so the report explicitly returns `scalar_factors_consumed = false`.

## Safe and Unsafe Paths

| Path | State | Reason |
|---|---|---|
| Finite SMAO whitened Q/K → linear attention Q/K | Supported | Geometry-preserving coordinate preprocessing; O(sequence × hidden) copy |
| SMAO scalar factors → linear feature map | Not silently supported | Semantics are not proven equivalent |
| Dense SMAO Gaussian pairwise scores | Rejected by project invariant | Would introduce O(sequence²) pair work/storage |
| Failed or nonfinite SMAO output → linear path | Rejected | Upstream numerical failure cannot be hidden |

The downstream linear recurrence remains `O(batch × sequence × hidden²)` arithmetic with `O(batch × hidden²)` streaming state. The adapter allocates no sequence-by-sequence matrix.

## Verification

The native Release suite passed **59/59 tests**. The portable AddressSanitizer and UndefinedBehaviorSanitizer suite also passed **59/59 tests**, with no sanitizer findings or compiler diagnostics.

Boundary tests verify successful coordinate adaptation, explicit non-consumption of scalar factors, composition with `LinearCausalAttention`, rejection of failed SMAO status, rejection of nonfinite whitened coordinates, shape preservation, and finite outputs. The live-source complexity scan found no `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` token-pair allocation pattern.

## Remaining Scope

The boundary is intentionally conservative. It does not claim that linear feature-map attention reproduces exact SMAO Gaussian decomposition or dense softmax attention. The next transformer work is feed-forward and normalization structure, followed by output processing and configuration serialization; any future SMAO integration must preserve this explicit semantic boundary.
