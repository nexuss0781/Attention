# Attention Performance Progress Report

**Current revision:** `0cb537b`  
**Baseline revision:** `2bc0125`  
**Measurement host:** six-vCPU Intel Xeon virtual CPU, Ubuntu 24.04, GCC 13.3, Release `-O3 -march=native`, OpenMP linked.  
**Test status:** 126/126 Release tests passing.

## Implemented upgrades

| Commit | Upgrade | Effect |
|---|---|---|
| `1e12d30` | Shared prefix scan in analytical backward; periodic validation in the training executable | Removes the previous per-position prefix rebuild and reduces validation multiplier |
| `9d9b63e` | Optional global gradient clipping and Module 1 default clip of 1.0 | Adds a bounded update norm for stability |
| `7bc211c` | Eigen row-major GEMM for QKV, FFN, and vocabulary projections | Replaces scalar dense forward loops |
| `56f27ef` | Configurable batch size and cached sinusoidal positional encoding | Enables future batched updates and removes repeated trigonometric work |
| `0cb537b` | Exact parameter-count and memory-accounting correction | Makes reported capacity match registered checkpoint tensors |

## Before/after measurements

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| Full model forward, context 32 | 8,767.61 tokens/sec | 38,352.8 tokens/sec | 4.37x faster |
| Real training with 8K validation after every step | 22.587 training tokens/sec | Not used after repair | Invalid operational policy |
| Training with periodic validation, minimal validation, shared backward scan | 45.372 tokens/sec before GEMM | 132.179 tokens/sec with final kernels and 8K validation policy | 2.91x versus periodic pre-GEMM path |
| Training with minimal validation, batch 1 | 87.761 tokens/sec before reverse scan | 169.857 tokens/sec after reverse scan, clipping, GEMM, and batching control | 1.94x |
| Linear-attention kernel, hidden 32, sequence 100K | 296,420 tokens/sec | Same algorithmic kernel | Forward linear-attention invariant preserved |

The forward result is measured on the full 2-layer language model, not the isolated hidden-size-8 linear-attention microbenchmark. The training result includes analytical backward, gradient computation, logging, checkpoint validation, and the current periodic validation policy. It is therefore the relevant operational metric.

## Complexity audit

The backward attention implementation now creates one prefix state per sequence position and updates it once in a forward scan. It no longer rebuilds all source prefixes for each output position. The core path contains no dense attention-score tensor, no `[sequence, sequence]` attention matrix, and no `source <= position` prefix-reconstruction loop. The linear-attention state remains dimension-bounded at inference time and sequence-linear in the forward scan.

The production path still contains a scalar tape and scalar backward projections, so the system is not yet at the intended final performance ceiling. The next major performance boundary is a tensor-level backward implementation with a reverse scan. The current shared prefix scan removes the hidden quadratic token-pair computation but still stores sequence-length prefix states for the backward graph.

## Quality and architecture status

The current model still uses a single linear-attention state despite a serialized `attention_head_count=4` field. True head-wise computation and an attention output projection remain unimplemented. The space-token collapse competency failure is therefore still a real quality failure; no new competency claim is made by these speed improvements.

The model parameter estimator now matches the registered tensors. For the Stage 0 checkpoint, the serialized trainable element count is 31,940; stale earlier estimates should not be used.

## Sanitizer result

The Release suite passes 126/126. A separate AddressSanitizer/UndefinedBehaviorSanitizer run built successfully, but one unrelated existing SMAO test, `WhitenCoordinatesTest.IsometryProperty`, failed deterministically with `Status::4` and a relative-error assertion. The failure occurs in `tests/test_whiten_coordinates.cpp`, not in the modified transformer, optimizer, training, or projection files. It should be resolved in a separate numerical-stability task before claiming a completely clean sanitizer gate.

## Next performance gate

The next implementation should add a production tensor backward and compare sequence-length scaling at T=32, 64, 128, and 256. The acceptance criterion is approximately linear backward growth, stable gradients, and no regression in the existing 126-test suite. Only after that should the project introduce AdamW with persisted moments, real batch training, true multi-head state mechanisms, and matched GRU/Transformer/SSM baselines.

## True multi-head state milestone

Commit `76751bb` makes the configured head count executable. With hidden size 32 and four heads, each head uses an 8x8 recurrent state rather than one 32x32 state. The four-head forward path, streaming path, and analytical backward path use the same partitioning. The Release suite increased to 127 tests, with 127/127 passing.

A matched microbenchmark on the same real token stream measured approximately 832K tokens/sec for four-head Attention, 2.15M–2.32M tokens/sec for the GRU comparator, 4.14M–4.53M tokens/sec for a diagonal SSM comparator, and 17.9K tokens/sec for the causal dense softmax Transformer at context 2,048. Attention therefore already dominates the dense Transformer at long context, but it does not yet dominate GRU or diagonal SSM raw throughput. This is an optimization target, not a result to conceal.

The current architecture comparison is documented separately in `docs/ARCHITECTURE_COMPARISON_AUDIT_2026-08-15.md`.
