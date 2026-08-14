# Stage 2.4 Checkpoint Reload Report

The Stage 2.4 checkpoint slice is complete through `attention::TransformerCheckpoint`. It serializes the validated configuration, frozen tokenizer metadata, and all finite model parameter values deterministically in `attention.checkpoint.v2`, reloads them into a fresh model, and enforces exact tokenizer identity, parameter name, shape, rank, and scalar-count agreement.

Reload equivalence was verified by initializing a source model, serializing it twice for byte-identical output, loading into a fresh model, and comparing full forward logits for the same token stream. Tokenizer metadata mismatch, malformed tokenizer metadata, malformed magic, embedded configuration, trailing data, duplicate parameter ordering, destination reuse, and nonempty destination stores are rejected.

Verification completed:

| Gate | Result |
|---|---:|
| Focused checkpoint tests | 5/5 passed |
| Full Release suite | 108/108 passed |
| Portable ASAN/UBSAN suite | 108/108 passed |
| Deterministic byte serialization | Passed |
| Exact forward-logit reload equivalence | Passed |
| Malformed checkpoint/configuration rejection | Passed |
| Forbidden token-pair complexity scan | Passed |

This completes the Stage 2.4 checkpoint reload and malformed-configuration checklist items. Checkpoint v2 now binds every model artifact to the frozen tokenizer metadata, while the per-run manifest records the same identity before training begins.
