# Long-Context Feasibility Under the Personal-Computer Baseline

**Assessment date:** 2026-08-14

**Repository:** `nexuss0781/Attention`

## Executive Decision

A **100-million- or 1-billion-token logical context** is architecturally feasible only as a streamed recurrent context. It is not feasible to keep the entire token activation sequence resident in the current personal-computer memory budget, and it is not feasible to provide exact arbitrary-token retrieval from a constant-size linear state without an additional external or hierarchical memory system.

The correct efficient architecture is therefore:

> **Chunked input and output tensors + per-layer recurrent linear-attention state + absolute-position offsets + optional external retrieval or hierarchical summaries.**

This is consistent with the recurrent interpretation of causal linear attention described by Katharopoulos et al. [1], while LongNet demonstrates a different route—dilated sparse attention—with a claimed billion-token sequence capability and linear computation [2]. Mamba demonstrates a further alternative based on input-dependent selective state-space recurrence and reports linear sequence scaling and results up to million-length sequences [3]. These references support the architectural direction, but none of them makes a billion-token context equivalent to retaining every token for exact arbitrary retrieval.

## Hardware Baseline

The current verification environment has an Intel Xeon processor at 2.50 GHz with 6 logical CPUs and approximately 3.8 GiB physical RAM available to the sandbox. Attention’s current implementation is F32 CPU-oriented, with no GPU runtime requirement.

## Resident-Memory Bound

For a sequence tensor with `N` tokens and hidden width `H`, one F32 tensor requires `4 × N × H` bytes. Keeping Q, K, V, and one output tensor resident requires approximately `16 × N × H` bytes before embeddings, intermediate projections, model weights, allocator overhead, or other layers.

| Logical window | One F32 tensor, H=64 | Four sequence tensors, H=64 | Interpretation |
|---:|---:|---:|---|
| 1,000,000 | 0.238 GiB | 0.954 GiB | Possible only as a carefully bounded single-layer resident workload on the baseline machine |
| 100,000,000 | 23.842 GiB | 95.367 GiB | Not resident in the baseline machine |
| 1,000,000,000 | 238.419 GiB | 953.674 GiB | Not resident in the baseline machine |

At hidden width 768, one F32 tensor at one billion tokens is approximately 2,861 GiB. Consequently, a 100M or 1B window must be treated as a **logical streamed window**, not as a tensor allocated at once.

## Recurrent State Bound

The implemented `LinearAttentionState` stores a per-batch matrix and normalizer of approximately `H² + H` F64 values plus two F32 feature scratch vectors. Its memory is independent of sequence length. For batch one, the state is approximately:

| Hidden width | Streaming state, batch one |
|---:|---:|
| 64 | 33 KiB |
| 768 | 4.512 MiB |
| 2048 | approximately 32 MiB |

For a multi-layer model, the recurrent state is multiplied by the number of layers. This remains much smaller than sequence-resident activation storage, but model weights, optimizer state, and per-token projection compute still dominate practical deployment costs.

## Measured Scaling

The dedicated chunked benchmark processed exactly one million logical tokens using 4,096-token chunks, hidden width 8, and a constant streaming state. The measured result was **306.496 ms**, approximately **3.263 million tokens/sec**, with **640 bytes** of reported recurrent state. No full one-million-token sequence tensor was allocated by the benchmark.

The implementation also passed a logical one-billion-token context test by configuring the stream with `context_length = 1,000,000,000` while allocating only dimension-bounded state. The test processes only a one-token prefix; it proves that context length itself does not force a context-sized allocation, not that a complete billion-token prefill is instantaneous.

A rough planning projection from the measured hidden-width-8 benchmark, assuming the current scalar recurrence’s `H²` arithmetic scaling, is:

| Hidden width | Projected 1M tokens | Projected 100M tokens | Projected 1B tokens |
|---:|---:|---:|---:|
| 64 | 19.3 s | 32.2 min | 5.36 h |
| 768 | 46.3 min | 77.2 h | 32.2 days |

These are **single-layer CPU planning projections**, not measured full-model results. A real multi-layer language model multiplies the compute by its layer count and adds projections, feed-forward layers, normalization, data movement, and tokenization. The current scalar kernel must be optimized substantially before 100M–1B prefill is operationally practical on a personal computer.

## Important Quality Limitation

Constant memory does not imply unlimited information capacity. A fixed recurrent state compresses the past. It can support streaming summarization and certain associative retrieval patterns, but exact arbitrary recall of every token is not guaranteed. For a useful 100M–1B context system, Attention should combine the recurrent state with one or more of the following:

| Mechanism | Purpose | Cost |
|---|---|---|
| Hierarchical summaries | Preserve multi-scale semantic information | Additional model and training design |
| Chunk-level memory | Keep selected representations outside the active recurrent state | Disk/RAM storage and retrieval policy |
| Sparse or dilated links | Preserve selected long-range access paths | More complex scheduling and possible non-linear factors |
| External retrieval index | Recover exact source spans when needed | Index build, storage, and query latency |

The architecture must not claim that a single constant-size linear state is a drop-in replacement for full exact attention. The current contract therefore supports the efficient stream while leaving retrieval and hierarchical memory as explicit future modules.

## Current Implementation Changes

The repository now contains `LinearAttentionState`, which supports chunked append operations, maintains `tokens_processed`, exposes sequence-independent `state_bytes`, and enforces the configured logical context limit. Sinusoidal positional encoding now provides `apply_at`, preserving absolute position across chunks rather than restarting positions at zero. These are the minimum foundations needed for a long logical window.

## References

[1]: https://proceedings.mlr.press/v119/katharopoulos20a.html "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"

[2]: https://arxiv.org/abs/2307.02486 "LongNet: Scaling Transformers to 1,000,000,000 Tokens"

[3]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
