# Stage 0 Training Result

The minimal bounded Stage 0 training loop is operational on the fixed four-token causal sequence `0,1,2,0` using the existing tiny configuration (`vocabulary_size=3`, `context_length=4`, one layer, hidden size 2, one attention head, feed-forward size 4). Parameters are initialized deterministically with seed 17 and updated by finite-safe SGD.

The executable `attention_stage0_training` performs four `Trainer::step` calls, logs loss before and after each update, serializes the updated checkpoint, reloads it into a fresh model and empty parameter store, and verifies the reloaded loss. The captured trace is stored in `/tmp/attention_stage0_training.csv` for this run.

Release verification completed successfully: all 112 tests passed, including the four new Trainer tests. The focused trainer checks cover a strict one-step loss decrease, finite updated parameters, checkpoint reload loss equivalence, invalid batch shapes, nonfinite learning rates, and nonfinite gradients. The targeted complexity scan found no `n*n`, `sequence_length*sequence_length`, `context_length*context_length`, or `token_count*token_count` pattern.

This is a training-signal validation run, not a language-quality claim. It intentionally uses the tiny Stage 0 fixed subset and does not claim performance on FineWeb2, Amharic Wikipedia, OASST1, Aya, PG-19, or long-context quality.
