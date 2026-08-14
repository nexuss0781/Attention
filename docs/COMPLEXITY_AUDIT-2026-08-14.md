# Attention Complexity and Allocation Audit

**Audit date:** 2026-08-14

**Branch:** `master`

## Conclusion

The production Attention implementation does **not** introduce an O(n²) token-sequence computation or allocation. The current paths scale linearly with the number of tokens for fixed hidden dimension, with additional work dependent on the model dimension `d`:

| Subsystem | Token scaling | Dimension scaling | Token-pair allocation |
|---|---:|---:|---|
| Input validation | `O(n d)` | — | None |
| Exact decomposition | `O(n d)` | — | None |
| Whitening | `O(n d²)` for the row-wise kernel or equivalent GEMM | `O(d²)` operator | None |
| Forward output buffers | `O(n d)` plus `O(n)` vectors | `O(d²)` metric/operator | None |
| Token embedding | `O(batch × sequence × hidden)` | — | None |
| Sinusoidal positions | `O(batch × sequence × hidden)` | — | None |
| Metric assembly | — | `O(d²)` storage and dimension-dependent eigensolver work | None involving `n` |
| Anisotropic distance | `O(d)` per query/key pair explicitly requested by the caller | `O(d)` | No batch pair matrix |

The only current `n × n` allocation found in the source tree was in `tests/test_exact_decomposition.cpp`, where the algebraic reference test materialized two score matrices. That was test-only rather than production behavior, but it contradicted the project’s memory-bounded verification policy. The test has been repaired to stream each pairwise reference comparison with constant extra memory. The reference comparison remains mathematically exhaustive over pairs, so its **time** is still `O(n² d)` by design; it no longer has `O(n²)` memory.

## Source Evidence

The production exact-decomposition loop iterates over rows and then dimensions. The production whitening path iterates over rows and dimensions, or calls row-major matrix multiplication with shapes `n × d`, `d × d`, and `n × d`. The forward path allocates `n × d` whitened outputs, `n` scale vectors, and `d × d` metric/operator matrices. The new embedding and positional paths iterate over batch, sequence, and hidden dimensions only.

No production source path contains a loop over token index `i` nested with a second token index `j` to construct an attention-score matrix. No production allocation uses `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` storage.

## Runtime Scaling Evidence

The strict native benchmark was run with `d=64` and three warmed samples. The relevant p95 results were:

| Workload | `n=10,000` | `n=100,000` | `n=1,000,000` |
|---|---:|---:|---:|
| Decomposition p95 time | 1.414 ms | 15.590 ms | 153.562 ms |
| End-to-end p95 time | 2.214 ms | 7.875 ms | 98.117 ms |

Increasing `n` by ten produces approximately tenfold decomposition time from `10⁴` to `10⁵` and again from `10⁵` to `10⁶`, which is consistent with linear token scaling for the tested fixed `d`. End-to-end results include fixed overhead, cache, and threading effects, but do not show the quadratic growth that would be expected from an `n × n` production score matrix.

A process-level RSS sample also showed approximately **130 MiB** peak for the non-full benchmark range through `n=100,000` and approximately **1.28 GiB** for the full range through `n=1,000,000`. The latter is consistent with the explicitly allocated `Q`, `K`, `V`, whitened `Q`, and whitened `K` buffers, all scaling as `O(n d)`, rather than an `n × n` score matrix.

## Archive Boundary

`Attention.zip` is a tracked historical archive from 2026-06-05 and is not consumed by CMake, tests, or benchmarks. It contains an older copy of the algebraic-equivalence test with the two obsolete `n × n` reference buffers, as well as archived build artifacts. The current source tree and build use the live files on `master`; the archive should be regenerated or removed before it is presented as a current release artifact.

## Verification After the Fix

The Release configuration passed **50/50 tests** after converting the reference test to streaming comparison. The production benchmark remained strict-valid and met its applicable p95 targets. No source-tree `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` allocation remains outside the historical archive.
