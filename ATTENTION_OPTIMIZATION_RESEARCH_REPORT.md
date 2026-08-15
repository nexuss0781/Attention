# Attention Optimization and Sequence-Architecture Research Report

**Repository audited:** `nexuss0781/Attention`  
**Audited revision:** `2bc0125230679d0e399c794b0428f95dbd009d53`  
**Date:** 2026-08-15  
**Scope:** C++20 implementation, current Stage 0 language-model path, linear-attention kernel, analytical backward pass, training loop, data path, benchmarks, and comparison with standard Transformers, GRUs, S4/SSM, Mamba, and related linear-attention systems.

> **Executive conclusion:** The current Attention project is not blocked primarily by the forward linear-attention asymptotic. It is blocked by the **training implementation around it**: a scalar reverse-mode tape, a hidden quadratic prefix reconstruction in the backward attention graph, batch size one, full validation on every training step, repeated full-sequence forward passes, scalar dense projections, and an optimizer that is too weak for stable language-model training. The forward model can process the current 32-token window at approximately **8,768 tokens/sec** on the audit CPU, while the real training path falls to approximately **22.6 training tokens/sec** when the 8K validation set is evaluated after every step.

## 1. Audit basis and reproducibility

The repository is a C++20 CPU implementation built with CMake, Eigen, optional OpenMP, and an optional CBLAS configuration. The audit used the existing Release build at `/tmp/attention_stage0_build`, compiled with `-O3 -march=native`, on a six-vCPU Intel Xeon virtual machine with AVX2/AVX-512 features exposed. The benchmark results below are **sandbox CPU measurements**, not Colab GPU measurements and not claims about a production deployment.

The repository test suite completed with **125/125 tests passing**. This confirms the existing functional tests, checkpoint tests, gradient tests, streaming-state tests, and long-context logical-state tests, but it does not establish language quality, efficient training, or absence of hidden quadratic work in the analytical backward graph.

The current Stage 0 checkpoint configuration is vocabulary 260, context 32, two layers, hidden size 32, four configured attention heads, FFN size 128, tied embeddings, causal mode, and F32 precision. The serialized checkpoint contains **31,940 trainable scalar values** and 32 parameter tensors. The configuration estimator returns **35,140**, a discrepancy of 3,200 values. The estimator counts a positional-encoding parameter block that is not stored and counts four hidden-by-hidden matrices per layer even though the executed QKV implementation stores three. This is a correctness/documentation issue in the capacity accounting and should be fixed before comparing model sizes.

## 2. Current measured throughput

### 2.1 Measured results

| Path | Configuration | Measurement | Result | Interpretation |
|---|---|---:|---:|---|
| Full model forward | 2 layers, hidden 32, context 32, vocab 260, batch 1 | 1,000 repeated forwards after warm-up | **8,767.61 tokens/sec** | Forward-only model throughput; no backward, no validation, no checkpointing |
| Validation loop | Same model, 8,000 validation tokens, 250 windows of 32 | One complete `ValidationEvaluator` pass | **8,609.65 tokens/sec** | Includes batch copies, full forward, full vocabulary loss, and repeated allocations |
| Training loop, real Module 1.1 validation | 8 training steps, 32 tokens/step, 8,000 validation tokens every step | End-to-end executable wall time | **22.587 training tokens/sec** | Includes loss-before, analytical backward, SGD, loss-after, full validation after every step, logging, and final checkpoint checks |
| Training loop, minimal validation | Same model and 8 steps, 32 validation tokens | End-to-end executable wall time | **87.761 training tokens/sec** | Approximate upper bound for the current scalar training path before large validation overhead |
| Linear attention operator | Hidden 8, sequence 1,000,000 | Existing p95 benchmark | **3.072M tokens/sec** | Core operator only, not the language model |
| Streaming linear attention | Hidden 8, one million logical tokens, 4,096-token chunks | Existing streaming benchmark | **3.398M tokens/sec** | State update only, not embeddings, blocks, FFN, logits, or training |
| Linear attention sweep | Hidden 32, sequence 100,000 | One forward pass | **296,420 tokens/sec** | More relevant to the current hidden size; operator only |
| Linear attention sweep | Hidden 64, sequence 10,000 | One forward pass | **85,775 tokens/sec** | Shows the quadratic dependence on hidden width inside the nominally linear-in-sequence operator |
| Linear attention sweep | Hidden 128, sequence 10,000 | One forward pass | **21,938 tokens/sec** | State and update cost become dominant as hidden width grows |

The direct linear-attention implementation is linear in sequence length for fixed hidden size, but its dominant term is approximately **O(T·H²)**. The measured sweep makes this visible: increasing hidden size from 32 to 64 reduces throughput from about 296K to 86K tokens/sec, and increasing it to 128 reduces throughput to about 22K tokens/sec. The fixed state is only H² plus H values, but the per-token update still performs an H-by-H outer-product accumulation and an H-by-H query-state contraction.

### 2.2 Projected Module 1 wall-clock cost

The current controller evaluates the complete validation set after **every training step**. Using the measured 8,609.65 validation tokens/sec and the measured minimal-validation training core, the following are approximate projections for one pretraining plus the configured fine-tuning pass. They assume the current CPU behavior remains linear with validation size; actual Colab values will vary.

| Submodule | Train steps | Validation tokens per step | Approx. pretraining time | Approx. fine-tuning time | Approx. combined time |
|---|---:|---:|---:|---:|---:|
| 1.1 | 2,048 | 8,000 | 0.74 h | 0.09 h | **0.83 h** |
| 1.2 | 3,072 | 32,000 | 3.48 h | 0.44 h | **3.92 h** |
| 1.3 | 4,096 | 64,000 | 8.87 h | 1.11 h | **9.98 h** |
| 1.4 | 6,144 | 128,000 | 26.00 h | 3.25 h | **29.24 h** |

These projections explain the termination and training ineligibility symptoms. The controller is not merely training on the requested number of tokens. It repeatedly reprocesses the validation set inside the critical path. Module 1.4 would cause roughly 786 million validation-token evaluations before counting retries, and its configured work is not appropriate for an interactive Colab session without periodic validation, checkpoint intervals, and resumable chunking.

### 2.3 What the current numbers do and do not prove

The approximately 3.1M tokens/sec linear-attention result is a **kernel microbenchmark at hidden size 8**. It must not be presented as the language-model throughput. The approximately 8.8K tokens/sec full-model forward result is a much more relevant inference baseline for the current model, but it is measured over full 32-token windows and does not include autoregressive generation with a state cache. The approximately 22.6 tokens/sec training figure is the most important operational number because it measures the actual path that caused the run to become impractical.

## 3. Codebase findings by execution path

### 3.1 Forward architecture

`TransformerModel::forward` allocates an embedding tensor, a positional-encoding tensor, one output tensor per transformer block, a final normalization tensor, and a full `[batch, sequence, vocabulary]` logits tensor. Each block allocates normalized attention, Q/K/V tensors, attention output, residual output, normalized FFN, FFN output, and the final residual tensor. This is correct but allocation-heavy for a small CPU kernel and prevents reuse of workspaces across steps.

The configuration field `attention_head_count` is validated and serialized, but it is not used to split Q/K/V into heads, apply per-head attention, or combine heads with an output projection. The executed block therefore is **not multi-head attention** despite the configuration saying four heads. It is a single hidden-width linear-attention state. The architecture should either implement true head-wise computation or rename the field and document the current single-state design.

The block has Q, K, and V projections but no standard attention output projection. This reduces expressiveness compared with the usual Transformer block. It also means that increasing the configured head count currently does not create additional representational paths.

### 3.2 Linear-attention forward kernel

The forward kernel maintains a double-precision state of shape H×H and a normalizer of shape H. For every token, it computes positive exponential features, updates the state with a key outer product and value vector, computes a query-normalizer denominator, and contracts the query with the state. There is no dense token-pair matrix, so the forward path does not allocate an O(T²) attention matrix.

However, the kernel uses scalar nested loops and `std::exp` for every feature. It allocates a new double state and temporary feature vectors on every full forward call. It also resets and fills the full state for each batch. The use of double accumulators with F32 model tensors improves numerical safety but prevents the most direct F32 SIMD/FMA path and increases memory traffic.

The state memory is bounded by approximately `batch × (H² + H) × sizeof(double)`, which is excellent for long-context inference at small H. At H=32, the measured state is 8,448 bytes for batch one; at H=128 it is 132,096 bytes. The important limitation is not state memory alone: the state is a finite-rank associative memory, so it cannot preserve arbitrary exact token identity over 100 million positions. Long-context feasibility and long-context language quality are different claims.

### 3.3 Hidden quadratic work in analytical backward

The most serious architectural performance finding is in `analytical_backward.cpp`. The forward kernel is linear in sequence length, but the backward graph builder reconstructs the prefix state independently for every position:

```text
for position in sequence:
    initialize normalizer and state
    for source <= position:
        rebuild key features and key/value state
```

That is **O(T²·H²) tape construction** for the attention portion. It does not allocate a dense `[T,T]` matrix, so the repository’s explicit dense-matrix invariant is not violated literally, but it does reintroduce quadratic token-pair computation in the training backward path. This is the critical hidden scaling problem for the stated long-context objective.

The tape stores one scalar `Node` per primitive operation. Each node contains a double value, two integer parent IDs, and two double local derivatives, with approximately 40 bytes of storage before vector capacity overhead and the separate reverse-gradient array. The backward pass also creates `std::vector<std::vector<Var>>` structures for every token, Q, K, V, normalized state, FFN state, and loss. This creates many heap allocations and pointer indirections. It is mathematically clear and useful for small gradient verification, but it is not a production language-model training engine.

The backward path also reconstructs the final normalization and vocabulary projection for every prediction position as a scalar tape graph. At vocabulary 260 and hidden 32 this is already expensive; at a realistic vocabulary of 32K or 50K, the tape and projection cost would become prohibitive.

### 3.4 Training-loop overhead

`Trainer::step` computes a complete causal loss before the update, runs the complete analytical backward pass, applies SGD, and computes a second complete causal loss after the update. Thus every training step already contains two full forward/loss evaluations plus backward.

`benchmark_stage0_training.cpp` then runs `ValidationEvaluator::evaluate` after every step. The evaluator rewinds the validation loader and evaluates every validation batch. This is the single largest avoidable multiplier in the current run. Validation should be periodic, asynchronous where possible, and always performed at the end of a resumable segment. The current loop also writes a CSV line and appends JSON logging every step.

The loader uses batch size one and copies every 32-token window into a new vector. There is no shuffle, document-aware sampling, prefetching, memory mapping, zero-copy batch view, or batch accumulation. The deterministic contiguous stream is useful for a diagnostic lesson but weak for language-model optimization and can amplify local token-distribution collapse.

### 3.5 Dense kernels and numerical checks

QKV projection, feed-forward projection, and vocabulary projection are all scalar triple loops with double accumulators. The QKV path performs three separate matrix-vector-style passes. The FFN performs an up projection, GELU, and down projection one token at a time. The vocabulary path produces the full logits tensor for every position, even when only the last position is needed for generation.

The CMake project links OpenMP when available, but the language-model kernels contain no OpenMP pragmas. The optional CBLAS path is used in the SMAO whitening code, not in the transformer QKV, FFN, or vocabulary projection paths. Therefore “OpenMP enabled” and “CBLAS available” do not currently mean that the language-model hot loops are parallel or BLAS-backed.

Most kernels repeatedly call `all_finite()` across inputs, parameters, and outputs. These checks are valuable in a debug mode but should be configurable. In a production training step they cause repeated full-array scans, especially because the parameter store also linearly searches parameter names by string and recomputes gradient norms by traversing every gradient element after the backward pass.

Positional encodings recompute sine and cosine values on every forward call rather than using a cached table. This is a secondary cost at context 32 but becomes unnecessary work for longer windows.

### 3.6 Optimization and stability limitations

The current optimizer is plain elementwise SGD. It has no momentum, Adam-style adaptive moments, decoupled weight decay, gradient clipping, learning-rate warm-up, cosine schedule, loss scaling, or gradient accumulation. SGD can be useful as a correctness baseline, but it is not a strong default for training a byte-level language model through a randomly initialized transformer-like stack.

The F32-only configuration and double-based scalar accumulation are inconsistent: the model stores F32 values but the critical computation is neither a fast pure-F32 kernel nor a deliberate mixed-precision design. The correct production choice is either optimized F32 kernels with compensated accumulation where needed or a planned BF16/FP16 compute path with F32 master weights and loss scaling.

## 4. Diagnosis of the space-token collapse

The observed Module 1.1 result was not a simple lack of data. All 16 cases produced a single repeated space token, with `unique_generated_tokens=1` and `max_repeated_run=16`. This is a genuine competency failure.

The immediate causes are interacting:

| Cause | Evidence | Severity |
|---|---|---:|
| Training throughput is extremely low | 22.6 tokens/sec with real validation; full Module 1.1 projected near 0.83 hours including fine-tune | Critical |
| Validation runs after every step | 8K tokens are reprocessed 2,048 times | Critical |
| Analytical backward is scalar and tapes every primitive | Dynamic graph, per-scalar nodes, many heap allocations | Critical |
| Backward attention rebuilds prefixes for every position | Hidden O(T²·H²) training work | Critical for scale |
| Batch size is one | No gradient averaging across varied windows | High |
| SGD only | No adaptive normalization or clipping | High |
| Byte-level model with only 32-token context | High sample complexity and weak semantic signal | High for language quality |
| Four heads are not actually implemented | Capacity and inductive-bias mismatch | High |
| Greedy 16-token test from a weak checkpoint | Exposes collapse but does not explain it alone | Medium |
| Vocabulary is only 260 | Not the cause of collapse; it is actually computationally convenient | Low |

The loss decreasing from about 4.00 to 2.99 in the reported run does not prove sequence competence. A model can lower average cross-entropy by favoring common bytes such as spaces while failing every continuation test. The current competency gate correctly rejects the run.

The right interpretation is: **the training objective is receiving enough signal to change weights, but the implementation and optimization regime are not delivering a usable conditional distribution.** Before scaling the dataset, the training path should be repaired and benchmarked with a tiny synthetic memorization task, a deterministic byte-transition task, and a held-out continuation task.

## 5. Comparison with alternative sequence architectures

| Architecture | Training time shape | Inference state | Content-based retrieval | Long-context behavior | Strengths | Weaknesses relative to Attention |
|---|---|---|---|---|---|---|
| Standard softmax Transformer | O(T²·H) attention interaction work and O(T²) score memory in the naive form | KV cache grows with generated context | Strong exact token-to-token retrieval | Expensive prefill and memory without IO-aware kernels | Excellent parallel training and content reasoning | Quadratic attention representation; needs fused kernels and cache management |
| GRU | O(T·H²), sequential across time | O(H) hidden state | Gated summary, not arbitrary retrieval | Constant state but information is compressed | Small memory, simple recurrent inference, mature optimization | Sequential training, finite memory, weaker parallelism and retrieval |
| Current Attention linear kernel | O(T·H²) forward, O(H²) state | O(H²) state per layer | Kernelized finite-rank key/value memory | Excellent asymptotic state memory for fixed H | No dense token-pair matrix; simple streaming state | Current positive feature map has finite memory capacity; H² cost; quality not established |
| S4 | Structured SSM scan/convolution with fixed-size state | Fixed-size learned state | Strong long-range dynamics, limited direct content selection | Designed for very long sequences | Efficient long-range modeling and stable structured dynamics | More complex parameterization and kernels; content selection is weaker than attention unless enhanced |
| Mamba | Input-dependent selective SSM with hardware-aware scan | Fixed-size recurrent state | Selective propagation/forgetting based on input | Linear sequence scaling | Brings input-dependent selection to SSMs and targets fast inference | Complex fused implementation; quality and speed depend strongly on optimized kernels |
| Mamba-2 / SSD-style hybrid | Structured semiseparable computation with recurrent/chunked forms | Fixed-size or chunked state | Bridges attention-like and SSM-like computation | Strong potential for long sequences | A principled bridge between attention and SSM computation | Requires a substantially more advanced kernel and training implementation |

The standard Transformer paper emphasizes parallel training and shorter dependency paths, while S4 and Mamba target the long-sequence cost and state-management problem [1] [3] [4]. The GRU literature establishes that gated recurrence can outperform simple tanh recurrence and that GRU can be comparable to LSTM on evaluated sequence tasks [2]. FlashAttention demonstrates that even when the algorithm remains exact quadratic attention, IO-aware tiling can provide large wall-clock gains; asymptotic complexity and hardware efficiency are separate dimensions [5]. Linearized attention can also be interpreted as a finite fast-weight memory, which is directly relevant to the current H×H state design [7]. These comparisons imply that Attention should not choose between “linear attention” and “Transformer” as a single ideological decision. The strongest path is a **hybrid**: optimized dense projections, a true multi-head or grouped state mechanism, a selective recurrent/SSM branch for long memory, and a small exact/local attention branch for high-value content retrieval.

## 6. Major optimization upgrades

### Priority 0: repair the training algorithm before more dataset scaling

**Replace the scalar tape with an analytic tensor backward pass.** Keep the tape implementation only as a reference gradient oracle for tiny dimensions. The production backward should store compact layer activations and compute matrix/tensor gradients using blocked kernels. This one change removes the per-scalar node graph, most heap allocation, and the huge reverse traversal. It follows the same principle as IO-aware attention optimization: reduce unnecessary materialization and move computation through cache-friendly tiles rather than scalar graph nodes [5].

**Derive a reverse scan for linear attention.** The forward recurrence is a prefix state update. Its backward must use a reverse recurrence over the state rather than rebuilding every prefix for every output position. The implementation must be tested with a small finite-difference oracle and a long-sequence complexity test. The acceptance condition is that doubling T approximately doubles backward time, not quadruples it.

**Move validation out of the inner step loop.** Add `validation_interval_steps`, defaulting to a value such as 128 or 256, and always validate at the end of a segment. Preserve a validation report at each checkpoint, but do not recompute the same 8K, 32K, 64K, or 128K tokens thousands of times. This is the most immediate change for making Colab training eligible.

**Use a real batch.** Add batch size 16–64 where memory permits, use gradient accumulation when it does not, shuffle windows by document or deterministic seed, and keep document boundaries visible. The current sequence loader can remain as a diagnostic mode, but it should not be the production language-model mode.

### Priority 1: replace scalar dense kernels with blocked GEMM and fused operations

Fuse QKV into one projection where possible and use Eigen mapped matrix operations or a recognized CBLAS provider. The current three scalar matrix-vector passes should become one matrix-matrix operation over a batch of tokens. Apply the same change to FFN up/down projections and the vocabulary projection. Use F32 SIMD/FMA kernels first; retain a slower numerically guarded debug mode.

Fuse layer normalization, linear-attention feature transformation, residual addition, and output writes where practical. Cache the sinusoidal positional table. Reuse tensor workspaces across layers and steps rather than allocating fresh vectors for every operation.

### Priority 2: improve optimization and numerical stability

Implement AdamW or a comparable adaptive optimizer with gradient clipping, warm-up, and a schedule. Add gradient accumulation and explicit loss scaling if mixed precision is introduced. Add a debug-only `all_finite()` mode and a lower-frequency health-check mode for production. Record gradient norms and update norms, but avoid a full extra scan when it is not requested.

Fix the configuration accounting. `parameter_count()` must match the actual registered parameter tensors. Either add the missing output projection and true head parameters or revise the estimator to the implemented architecture. A model-size report that disagrees with the checkpoint undermines every throughput and capacity comparison.

### Priority 3: implement the stated architecture instead of metadata-only heads

Implement true multi-head linear attention with per-head projections and a head-combination projection, or explicitly rename the current block as single-state kernelized attention. A head-wise design can keep the total H² state cost if the state is partitioned carefully, while improving specialization and reducing the semantic mismatch between the configuration and the execution.

Consider a gated delta-rule or selective-update mechanism. Pure additive outer-product memory can accumulate stale associations. A learned write/update gate or delta correction can allow the model to overwrite incorrect key/value associations instead of only adding more state.

### Priority 4: make long-context quality a separate subsystem

For 100-million-token logical contexts, bounded state alone is not equivalent to remembering exact details. Add a hierarchical, trainable memory system with fixed-size segment summaries, selective state updates, and a bounded retrieval mechanism. The retrieval path must use fixed-size top-k or hashed routing and must not create all-pairs token matrices. The existing hierarchical-summary code should be treated as an infrastructure primitive, not as proof of learned long-context language quality.

A strong experimental branch is a hybrid block: local exact attention over a short window, a selective SSM or recurrent state for long-range continuity, and the current linear state as a content-addressable compressed memory. The SSM/attention connection is not merely conceptual; structured semiseparable formulations provide a mathematical bridge between recurrent state updates and attention-like computation [6]. Compare this against pure linear attention on the same data, parameter count, and wall-clock budget.

## 7. Recommended implementation order and acceptance gates

| Stage | Change | Required measurement | Acceptance gate |
|---|---|---|---|
| A | Add a benchmark harness for forward, loss, backward, validation, and checkpoint operations | tokens/sec, ms/step, peak RSS, allocations | Baseline reproducible on one command |
| B | Periodic validation and persistent segment checkpoints | wall-clock per training token | Module 1.1 projected work reduced by at least an order of magnitude |
| C | Tensor analytic backward with linear-time prefix reverse scan | backward tokens/sec and complexity sweep | 2x sequence length does not cause 4x time growth |
| D | Batched loader, gradient accumulation, shuffle, AdamW, clipping | loss curve and tokens/sec | Synthetic byte-transition task memorizes reliably without collapse |
| E | GEMM/fused QKV, FFN, and logits | forward and train tokens/sec | At least 5x training throughput improvement before increasing data budget |
| F | True multi-head or explicitly redesigned state mechanism | held-out continuation and ablation quality | Non-degenerate generation on at least 80% of Module 1.1 cases |
| G | Selective state/hybrid memory | long-range retrieval, forgetting, and throughput | Long-context quality improves without O(T²) allocations |

The first quality gates should be small and deterministic. Before returning to FineWeb, the model should learn a synthetic byte transition exactly, then a small repeated-text continuation task, then the 64K/8K Module 1.1 lesson. If the synthetic task fails after the analytic backward and optimizer changes, the issue is architectural or implementation-level rather than dataset scale.

## 8. Final assessment

Attention has a valid and useful core idea: a bounded-state linear attention operator that avoids a dense token-pair matrix and can stream over very long logical sequences. The implementation currently demonstrates that core operator at millions of tokens/sec for a tiny hidden width. That result is meaningful but insufficient for the project goal.

The present language-model path is not yet an efficient continual-learning system. The production backward pass secretly reconstructs quadratic prefixes, the trainer repeatedly validates the entire validation corpus, the dense layers are scalar and allocation-heavy, and the optimizer is a baseline SGD loop. These issues explain why a small 34K-scale model can terminate or become ineligible before a meaningful training session completes.

The correct next move is **not** to increase the dataset or context again. The correct move is to build a measured optimization baseline, remove the hidden quadratic backward and per-step validation, replace the tape with a tensor backward, introduce batching and AdamW, and only then return to competency training. The no-dense-attention invariant should be strengthened to explicitly forbid not only O(T²) token-pair storage but also O(T²) token-pair computation anywhere in forward, backward, validation, or evaluation.

## References

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"

[2]: https://arxiv.org/abs/1412.3555 "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"

[3]: https://arxiv.org/abs/2111.00396 "Efficiently Modeling Long Sequences with Structured State Spaces"

[4]: https://arxiv.org/abs/2312.00752 "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

[5]: https://arxiv.org/abs/2205.14135 "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

[6]: https://arxiv.org/abs/2405.21060 "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"

[7]: https://proceedings.mlr.press/v139/schlag21a.html "Linear Transformers Are Secretly Fast Weight Programmers"

[8]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/CMakeLists.txt "Attention CMake configuration"

[9]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/src/analytical_backward.cpp "Attention analytical backward implementation"

[10]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/src/linear_attention.cpp "Attention linear-attention forward implementation"

[11]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/benchmarks/benchmark_stage0_training.cpp "Attention Stage 0 training executable"

[12]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/src/validation.cpp "Attention validation evaluator"

[13]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/src/qkv_projection.cpp "Attention QKV projection"

[14]: https://github.com/nexuss0781/Attention/blob/2bc0125230679d0e399c794b0428f95dbd009d53/src/vocabulary_projection.cpp "Attention vocabulary projection"
