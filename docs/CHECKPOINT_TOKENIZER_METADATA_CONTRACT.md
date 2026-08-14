# Checkpoint Tokenizer Metadata Contract

Checkpoint format `attention.checkpoint.v2` stores the validated transformer configuration, a frozen tokenizer metadata record, and sorted finite parameter values. The tokenizer record contains the version string, vocabulary size, and BOS, EOS, PAD, and UNK IDs.

The default artifact is `attention.byte_utf8.v1` with vocabulary size 260 and special IDs BOS 256, EOS 257, PAD 258, and UNK 259. Metadata is framed and serialized deterministically before the parameter list. A checkpoint cannot be loaded unless its tokenizer metadata exactly matches the expected artifact metadata supplied to the loader. This prevents a model from silently consuming token IDs produced by a different vocabulary or special-token layout.

The metadata validation rejects empty or newline-containing versions, zero vocabulary sizes, out-of-range IDs, and duplicate special-token IDs. The checkpoint loader also requires a fresh model and empty parameter store, then validates configuration, parameter names, ranks, shapes, value counts, and finite values.

The format version is intentionally advanced from `attention.checkpoint.v1` because tokenizer metadata is now a required integrity field. Existing v1 payloads are rejected rather than guessed or silently upgraded. Every future training run must record the same tokenizer artifact identity in its run manifest and checkpoints.
