# Stage 0 Training Result

The minimal bounded Stage 0 training system now includes deterministic batch loading and structured per-step run logging. The executable `attention_stage0_training` consumes a flat token stream, packs it into fixed causal batches, runs finite-safe SGD, records losses and gradient norms, serializes the updated checkpoint, reloads it into a fresh model and parameter store, and verifies post-reload loss continuity.

The verified local debug stream contains four repetitions of the fixed causal sequence `0,1,2,0`, producing four batches with `batch_size=1` and `sequence_length=4`. Parameters use deterministic seed 17 and the existing tiny configuration (`vocabulary_size=3`, `context_length=4`, one layer, hidden size 2, one attention head, feed-forward size 4).

| Step | Loss before | Loss after | Gradient L2 norm | Tokens processed |
|---:|---:|---:|---:|---:|
| 0 | 1.18923914 | 1.16686773 | 0.679924309 | 4 |
| 1 | 1.16686773 | 1.14721239 | 0.636771500 | 8 |
| 2 | 1.14721239 | 1.12981844 | 0.598457456 | 12 |
| 3 | 1.12981844 | 1.11430740 | 0.564598024 | 16 |

The run log is deterministic across repeated executions when the same token stream, seed, configuration, and code-commit metadata are supplied. The reloaded checkpoint reproduces the final loss exactly at `1.11430740`. The structured log records the stage, dataset identifier and revision, tokenizer version and vocabulary size, architecture serialization, code commit, seed, batch shape, learning rate, batch offsets, processed-token counts, losses, and gradient norms.

Release verification completed successfully with `119/119` tests passing. Portable ASAN/UBSAN verification also completed with `119/119` tests passing. The live-source complexity scan over `src/`, `include/`, and `benchmarks/` found no targeted `n*n`, `sequence_length*sequence_length`, `context_length*context_length`, or `token_count*token_count` token-pair allocation pattern.

This remains a **training-signal and pipeline validation run**, not a language-quality claim. The local debug stream is synthetic and the repository's normalized measurement corpus is explicitly not training-approved. The next required training work is validation evaluation, configurable run control, and resumable optimizer/training state before scaling to the selected curriculum datasets.
