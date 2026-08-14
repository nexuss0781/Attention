# Hierarchical Summary Contract

`attention::HierarchicalSummaryState` provides a bounded long-context memory path for coarse multi-scale context. It accepts finite `[batch, sequence, hidden]` chunks and incrementally aggregates token-local means across a configurable number of levels. Each level groups `fanout` completed summaries from the previous level; completed means are propagated upward as a single vector.

The state retains the running partial sum and count for every batch/level, the latest completed mean for every batch/level, and one hidden-width carry buffer. It does not retain the token sequence, token IDs, attention scores, or any token-pair matrix. Resident state is bounded by `O(batch × levels × hidden)` memory, independent of logical context length and processed token count.

Snapshots return `[batch, levels, hidden]` summaries. A level exposes its latest completed mean when no group is currently partial; otherwise it exposes the mean of the current partial group. These summaries are **approximate coarse context features**, not an exact retrieval index and not a guarantee of token-level recall. The quality-comparison protocol remains required before making any long-context quality claim.

The state enforces positive context, batch, hidden, and level dimensions, `fanout >= 2`, matching input metadata, finite values, and logical-context bounds. Chunked append and single-pass append are required to produce identical summaries for the same token stream.
