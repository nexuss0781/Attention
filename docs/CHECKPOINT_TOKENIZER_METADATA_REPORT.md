# Tokenizer Metadata Checkpoint Report

Checkpoint serialization now uses `attention.checkpoint.v2` and includes the exact tokenizer artifact identity required to interpret token IDs. The default `attention.byte_utf8.v1` metadata records vocabulary size 260 and special-token IDs BOS 256, EOS 257, PAD 258, and UNK 259.

The serializer validates metadata before writing it. The loader validates the serialized record and compares it against the expected tokenizer metadata before registering model parameters. A deliberate custom tokenizer version can be serialized and reloaded only when the same metadata is explicitly supplied; loading it with the default artifact is rejected. Malformed vocabulary sizes and duplicate or out-of-range special IDs are rejected.

| Gate | Result |
|---|---:|
| Focused checkpoint tests | 5/5 passed |
| Deterministic metadata bytes | Passed |
| Exact forward-logit reload equivalence | Passed |
| Tokenizer metadata mismatch rejection | Passed |
| Malformed tokenizer metadata rejection | Passed |
| Existing malformed checkpoint and fresh-destination rejection | Passed |

Full Release and sanitizer verification are required before this slice is marked complete in Todo.md. Existing v1 checkpoints are intentionally rejected because the tokenizer metadata field is mandatory in v2.
