# Long-Context Quality Comparison Protocol

**Status:** Protocol defined; execution blocked until the model, tokenizer, training loop, checkpoint format, and real evaluation data are implemented.

## Purpose

This protocol prevents a context-window claim from being accepted because a configuration object accepts a large number or because a short smoke test remains finite. A 100-million-token window is considered successful only when the model demonstrates measurable language or retrieval quality at long distances and does not materially regress against an ordinary-transformer reference under matched conditions.

## Current Readiness Decision

The checked-out repository currently contains the SMAO numerical foundation and transformer architecture configuration. It does **not yet contain** a tokenizer, trainable transformer computation graph, causal language-model loss, backward propagation, optimizer, checkpoint format, real-data loader, or language-quality evaluator. Therefore, the repository cannot currently produce a valid perplexity, accuracy, generation-quality, or ordinary-transformer comparison. Any such number reported before those components exist would be non-comparable or fabricated.

This follows the project specification: a quality claim must include model, tokenizer, data, evaluation, sampling, compiler, hardware, and source-version metadata, and the base model must improve held-out validation loss over an untrained baseline.

## Matched Model Comparison

The comparison must use two models with the same tokenizer, vocabulary, parameter budget, hidden size, layer count, feed-forward size, initialization seed, training data, optimizer, training-token budget, precision, and checkpoint-selection rule. The only architectural variable should be the attention mechanism:

| Reference | Context role | Requirement |
|---|---|---|
| Ordinary transformer | Short-window quality reference | Exact causal attention, tested only at memory-safe windows such as 1K–4K tokens on the verification hardware |
| Efficient Attention path | Long-window candidate | Streaming linear aggregation with fixed state, chunked inputs, and absolute positions |

The ordinary transformer is not required to run at 100M tokens on the personal-computer baseline. It is the quality anchor at overlapping windows. A long-context candidate must first avoid regressing at those overlapping windows before its 100M behavior is considered.

## Real-Data Evaluation

The first language-quality corpus should be PG-19, whose official repository describes books extracted from Project Gutenberg, with separate train, validation, and test splits and minimal processing. PG-19 is suitable for long-document language modeling but is not a complete general-purpose corpus; its historical style and biases must be recorded. The second evaluation layer should use L-Eval, which contains 20 long-context subtasks, 508 documents, more than 2,000 human-labeled query-response pairs, and input lengths from 3K to 200K tokens [1].

The evaluation runner must use fixed, versioned tokenization and disjoint test data. It must report token-level negative log-likelihood or perplexity where the task supports it, exact-match or F1 for retrieval tasks, and human-reviewed or validated rubric scores for generation tasks. Each result must retain per-example scores, not only an aggregate.

## Required Quality Gates

A result is not accepted as high quality until all applicable gates pass:

| Gate | Required evidence | No-go condition |
|---|---|---|
| Training validity | Loss decreases on a tiny overfit set and held-out validation loss beats the untrained baseline | No learning or unexplained instability |
| Overlap quality | Efficient and ordinary models compared at identical short windows | Efficient path regresses beyond a predeclared tolerance without explanation |
| Long-range retrieval | Queries whose answer occurs at controlled distances, including 3K, 16K, 64K, 200K, and later 1M/100M positions | Accuracy collapses as distance increases or is not measured |
| Language quality | PG-19 held-out perplexity/NLL with bootstrap confidence intervals | Only qualitative samples or a single unreplicated score |
| Robustness | Multiple seeds, chunk sizes, batch sizes, and checkpoint reloads | Result depends on one seed or chunk layout |
| Numerical stability | No NaN/Inf, bounded recurrent state, monotonic token accounting, and exact chunk-boundary checks | Any silent overflow, position reset, or state corruption |
| Resource correctness | Peak RSS, per-layer state bytes, throughput, and I/O recorded at every length | A large context is only accepted because the test truncated or skipped tokens |
| Reproducibility | Commit, model, tokenizer, dataset, evaluation version, seed, hardware, and command recorded | Results cannot be rebuilt |

A useful acceptance policy is to predeclare the permitted quality delta before training. The project must not choose a favorable threshold after inspecting results. If no threshold is yet justified by a baseline study, report the paired delta and confidence interval and classify the result as **measured but not accepted**, not as a pass.

## 100M Evaluation Requirements

A 100M context claim requires all of the following:

1. The runner must consume all 100M tokens; it may use fixed-size chunks but may not silently truncate, reset state, or restart positions.
2. The candidate must preserve absolute position across chunks and record the final token count.
3. Long-range probes must place answers at multiple distances, including near the beginning, middle, and end of the logical window.
4. The result must include a quality curve versus distance, not only a final aggregate.
5. The candidate must be compared with the ordinary-transformer reference at every overlapping feasible window.
6. The report must separate recurrent-summary quality from exact external retrieval quality if an external memory index is used.

A constant-size state can make the logical window memory-light, but it cannot by itself guarantee exact arbitrary recall of every past token. Therefore, a 100M result that relies on hierarchical summaries or external retrieval must identify those components and measure their contribution separately.

## Execution Status in Todo.md

The protocol maps to the existing Todo order. The current repository must first complete Stage 2’s forward graph, Stage 2.4 loss and backward execution, Stage 3 tokenizer, Stage 4 dataset lineage, Stage 5 training/checkpointing, and Stage 6 learning proof. Only then can the Stage 7 language-quality and context-behavior gates be executed honestly.

## References

[1]: https://aclanthology.org/2024.acl-long.776/ "L-Eval: Instituting Standardized Evaluation for Long Context Language Models"

[2]: https://github.com/google-deepmind/pg19 "PG-19 Language Modelling Benchmark"
