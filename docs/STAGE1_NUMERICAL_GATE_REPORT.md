# Stage 1 Numerical Foundation Gate Report

**Stage:** 1 — Stabilize the Existing Numerical Foundation

**Verification date:** 2026-08-14

**Verification branch:** `master`

## Completion Summary

The existing SMAO Phase 1 numerical foundation is now buildable, testable, numerically guarded, and safe at its public C boundary. The implementation has one canonical numerical-guard module, a functioning metric-backed distance handle, explicit frozen-gate measurements, repeated-output lifecycle safety, and strict repeated-sample performance verification.

The current source changes add a structured `FrozenGateReport` while preserving the existing boolean `check_frozen_gate_criteria` compatibility function. The C API now releases any prior result before replacing it during a repeated forward call, preventing the previous output buffers and opaque distance handle from being leaked.

## Build and Test Evidence

| Verification | Configuration | Result |
|---|---|---|
| Release build | Native SIMD, OpenMP enabled, CBLAS disabled | Passed; static library, shared library, tests, and benchmark built without warnings or errors |
| Regression suite | Same Release configuration | **40/40 tests passed** |
| Sanitizer build | Debug, native SIMD disabled, OpenMP disabled, AddressSanitizer and UndefinedBehaviorSanitizer enabled | **40/40 tests passed**; no sanitizer findings |
| Strict benchmark | Native SIMD, OpenMP enabled, `--full --strict --repeats 5` | Passed; both applicable p95 targets met |
| Portable numerical path | Native SIMD and OpenMP disabled | Covered by sanitizer configuration and regression suite |

The two tests added after the Stage 0 baseline are `Phase1ForwardTest.StructuredFrozenGateReport` and `CApiTest.RepeatedForwardReleasesPreviousOutput`.

## Structured Frozen Gate

The gate is evaluated by `evaluate_frozen_gate` and returns both criterion booleans and measured values. The compatibility function `check_frozen_gate_criteria` returns the report’s aggregate `passed()` result.

| Criterion | Acceptance rule | Current implementation |
|---|---|---|
| Output status | `Status::OK` | Explicit report field |
| Dimensions | Nonzero `n` and `d`; metric and whitening matrices are `d × d` | Explicit report field |
| Finiteness | All output vectors and matrices, condition number, and bandwidth are finite | Explicit report field |
| Minimum eigenvalue | At least `LAMBDA_MIN_THRESHOLD = 1e-6` | Self-adjoint eigensolver measurement |
| Condition number | At most `CONDITION_NUMBER_MAX_DEFAULT = 1e4` | Explicit report field |
| Whitening residual | Relative `WᵀW` reconstruction residual at most `2e-4` | Explicit measured residual |

The report is tested on a valid forward result and is used by the benchmark validity path. A failed criterion cannot be hidden by a successful status value.

## Performance Evidence

The strict benchmark was run on the recorded Intel Xeon verification environment with five warmed samples per case.

| Workload | p95 measurement | Target | Result |
|---|---:|---:|---|
| Exact decomposition, `n=1,000,000`, `d=64` | 6.488 million tokens/sec | At least 5 million tokens/sec | Met |
| End-to-end Phase 1, `n=1,000,000`, `d=64` | 94.742 ms | At most 200 ms | Met |

The benchmark reports mean, p50, p95, minimum, maximum, validity, and target status. The strict option returns failure when an applicable target is missed. CMake registers the same strict command as `AttentionPerformanceRegression` when `ATTENTION_ENABLE_PERFORMANCE_TESTS=ON`.

## Allocation Audit

The allocation audit is a source-level and test-level review of the Phase 1 path. The forward output owns only linear token buffers and fixed-size dimension buffers:

| Allocation or workspace | Shape | Scaling |
|---|---:|---:|
| Whitened queries | `n × d` | `O(n d)` |
| Whitened keys | `n × d` | `O(n d)` |
| Query scales | `n` | `O(n)` |
| Key weights | `n` | `O(n)` |
| Metric and whitening matrices | `d × d` each | `O(d²)` |
| Temporary metric/eigendecomposition state | Dimension-dependent | `O(d²)` |
| Token-pair or attention-score matrix | Not allocated | None |

No `n × n` or other token-pair auxiliary buffer is allocated by the Phase 1 forward implementation. The reusable `phase1_forward_into` path preserves output workspace capacity between calls, and the C API regression test verifies safe repeated replacement of the externally owned result.

## Public C API Contract

The opaque `internal_handle` is an owned `DistanceHandle` containing the dimension and assembled metric. It is created only after a successful forward computation and is released with the owning output. Distance calls validate null pointers, handle dimensions, finite query/key values, and finite nonnegative output. Unsupported F16 and BF16 modes are rejected explicitly because no conversion implementation exists yet.

A zero-initialized output is required for the first call. Subsequent calls to `smao_phase1_forward` release the prior output safely before producing a replacement. Release is idempotent, and all failure paths leave the output safe to release.

## Numerical and Portability Evidence

The numerical implementation uses Eigen for matrix operations and does not require a BLAS/LAPACK runtime. The CBLAS option remains available but is disabled in the release evidence above because Eigen is the canonical row-major-safe path. Fast-math is not enabled, preserving NaN and infinity detection.

OpenMP pragmas are now guarded by `_OPENMP`, so the sanitizer and portable configurations compile without unknown-pragmas warnings when OpenMP is disabled. Native AVX-512 code remains conditionally compiled and the portable path remains available for correctness verification.

## Stage 1 Acceptance

Stage 1 is complete under the project’s current scope. The numerical foundation builds from a clean checkout, all targets build, all 40 tests pass in release and sanitizer configurations, the C API exposes no fabricated distance behavior, the frozen numerical gate produces a structured report, the no-quadratic-allocation invariant is documented and audited, and the applicable strict performance targets pass.

The transformer computation graph, tokenizer, dataset pipeline, training runtime, and inference runtime remain later-stage work and are intentionally not claimed as Stage 1 deliverables.
