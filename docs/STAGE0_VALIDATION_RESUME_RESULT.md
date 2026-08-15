# Stage 0 Validation and Resume Result

The Stage 0 training path now has a deterministic held-out validation evaluator and a separate `attention.training_checkpoint.v1` envelope. The original `attention.checkpoint.v2` model format remains unchanged and continues to contain model configuration, tokenizer metadata, and model parameters only. The training envelope adds run identity, dataset identity and revision, global step, processed-token count, next batch index, and learning rate.

The integrated run trains on four batches of the fixed debug stream and evaluates a separate two-batch held-out stream after every update.

| Step | Training loss before | Training loss after | Validation loss | Gradient L2 norm |
|---:|---:|---:|---:|---:|
| 0 | 1.18923914 | 1.16686773 | 1.33287907 | 0.679924309 |
| 1 | 1.16686773 | 1.14721239 | 1.31487286 | 0.636771500 |
| 2 | 1.14721239 | 1.12981844 | 1.29960346 | 0.598457456 |
| 3 | 1.12981844 | 1.11430740 | 1.28665280 | 0.564598024 |

Validation loss is finite and decreases across this diagnostic run. The evaluator computes a prediction-token-weighted mean, resets the loader before and after evaluation, does not mutate parameters, and reproduces the same result on repeated evaluation.

The resume gate was tested in two ways. First, the executable writes the training-state envelope to disk, reloads it, and reproduces the final model loss exactly: `1.11430740`. Second, the test suite trains four batches uninterrupted, trains two batches, serializes and reloads the training state, skips the two already-consumed batches, trains the remaining two batches, and verifies every final parameter and final loss with exact floating-point equality against the uninterrupted run.

Release verification completed successfully with `125/125` tests passing. Portable ASAN/UBSAN verification also completed with `125/125` tests passing. Repeated executable runs produced identical core traces, JSON logs, and persisted training-state checkpoints after excluding only the intentionally different output-path lines. The live-source complexity scan over `src/`, `include/`, and `benchmarks/` found no targeted quadratic token-pair allocation pattern.

This gate is sufficient to proceed to the next engineering step, but it is not yet a FineWeb2 quality result. The repository's current local debug stream remains synthetic/measurement-only. FineWeb2 Stage 1 must remain blocked until an approved, checksummed, tokenized subset and its held-out split are prepared under the existing data policy.
