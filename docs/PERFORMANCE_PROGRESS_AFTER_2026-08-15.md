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

## Final after-audit snapshot

At revision `28a7aa7`, the Release build passes **127/127 tests**. Shell syntax checks and Python compilation checks pass. The refined static audit finds no dense attention-score matrix, no `source <= position` prefix rebuild, and no forbidden token-pair allocation pattern in the core source, headers, or registered benchmarks.

The final matched sweep remains approximately flat for four-head Attention at 0.827–0.841M tokens/sec from context 32 through 2,048. The comparable GRU reaches 2.09–2.40M tokens/sec, the diagonal SSM reaches 4.23–4.53M tokens/sec, and the causal softmax Transformer falls from 5.53M tokens/sec at context 32 to 18.8K tokens/sec at context 2,048. The honest verdict is therefore: Attention has the desired long-context scaling and beats dense softmax attention at long context, but it does not yet dominate optimized GRU or diagonal SSM kernels in raw throughput.

## Persistent AdamW milestone

Commit `9028a33` added opt-in AdamW with bias correction, decoupled weight decay, and clipping. The subsequent checkpoint extension adds `attention.training_checkpoint.v2`, which stores optimizer identity, hyperparameters, step count, and first/second moment tensors. A fresh optimizer successfully imports the reloaded AdamW state in the stage-0 probe, and the checkpoint regression suite now passes **129/129 tests**.

`run.sh` continues to default to SGD until the higher-level session runner loads the v2 training checkpoint for interrupted-session continuation. AdamW can be selected explicitly with `ATTENTION_OPTIMIZER=adamw`; the optimizer state is now serializable and importable, but automatic Colab session recovery must still be wired to consume it before making AdamW the unconditional default.
