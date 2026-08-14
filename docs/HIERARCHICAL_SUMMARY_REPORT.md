# Hierarchical Summary Long-Context Report

The first bounded long-context memory enhancement is complete through `attention::HierarchicalSummaryState`. The component maintains multi-scale running means with configurable fanout and level count. Each completed lower-level group is propagated upward as one vector, while each level retains its latest completed summary and current partial aggregate.

This is a memory mechanism, not a quality claim. It gives the transformer a bounded coarse-context signal, but it does not provide exact token retrieval and does not establish selective recall at 100M–1B tokens. The existing quality-comparison protocol remains mandatory for any future quality evaluation.

The state accepts a logical context of one billion tokens in tests while allocating only batch/level/hidden-dependent buffers. It supports chunked append, exact chunk-vs-single-pass equivalence for its summary semantics, deterministic snapshots, finite-value checks, shape validation, context overflow rejection, and state-byte accounting.

Verification completed:

| Gate | Result |
|---|---:|
| Focused hierarchical-summary tests | 4/4 passed |
| Full Release suite | 93/93 passed |
| Portable ASAN/UBSAN suite | 93/93 passed |
| One-billion-token logical-context bounded-state test | Passed |
| Chunked versus single-pass summary equivalence | Passed |
| Forbidden token-pair complexity scan | Passed |

The next ordered work is Stage 2.4 forward and backward execution, beginning with a complete forward pass and causal language-model loss. External retrieval memory and sparse long-range links remain separate future options; this summary component must not be represented as exact retrieval.
