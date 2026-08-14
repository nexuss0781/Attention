# Attention Todo

## Purpose

This document converts the Attention objective into an ordered, implementation-ready work plan. Every stage contains prerequisites, concrete tasks, deliverables, verification requirements, and a completion gate.

Work must proceed in dependency order. A later stage must not be treated as complete merely because its code exists; it is complete only when its acceptance gate has been demonstrated and recorded.

> **Operating rule:** Build the smallest complete, measurable, and dependable system first. Preserve modularity, correctness, observability, reproducibility, and local operability at every stage.

## Source Alignment

The workspace did not contain a file named `Goal.md` when this document was prepared. This plan is therefore aligned with the available Attention objective document, the SMAO Phase 1 specification, the repository implementation, and the documented end-to-end flow.

The plan preserves the full objective: technical definition, transformer architecture, tokenizer, data, training, language learning, instruction following, teacher interface, tools, external context, efficiency, runtime, testing, safety, documentation, and reproducible release.

## Status Legend

| Mark | Meaning |
|---|---|
| `[ ]` | Not started |
| `[-]` | In progress |
| `[x]` | Completed and verified |
| `[!]` | Blocked or requires a decision |
| `[*]` | Optional investigation that must not block the main path unless promoted |

## Current Focus — Verified Performance Path

**Status: started and active.** The project is currently focused on the high-performance, numerically safe execution path through the existing Attention Phase 1 foundation. Optimization work has already begun and has produced a verified target-meeting implementation; this is not a future or unstarted path.

> **Focus rule:** Keep the current numerical path correct, fast, measurable, and reusable before expanding into broader model, training, or runtime layers.

### Completed performance-path work

- [x] Established a profiling baseline for metric assembly, whitening, decomposition, validation, and gate checks.
- [x] Added native SIMD build support and OpenMP row parallelism.
- [x] Added the cache-resident AVX-512 fused Q/K whitening microkernel for the target `d=64` workload.
- [x] Added vectorized decomposition and combined query/key norm processing.
- [x] Fused finite-input and output validity checks into the optimized whitening path.
- [x] Removed redundant full-buffer scans from the trusted forward path.
- [x] Replaced expensive runtime sample verification with a direct `WᵀW` residual check.
- [x] Added reusable-output execution through `phase1_forward_into` to eliminate repeated large allocations.
- [x] Updated the benchmark to measure warmed steady-state execution.
- [x] Added regression coverage for output-workspace reuse.
- [x] Documented optimization modes and benchmark semantics in the README.

### Verified performance record

| Measurement | Verified result | Target | Status |
|---|---:|---:|---|
| Decomposition, `n=1,000,000`, `d=64` | `151.79 ms`, `6.59 M tokens/sec` | At least `5 M tokens/sec` | [x] Met |
| End-to-end Phase 1, `n=1,000,000`, `d=64` | `58.84 ms` | At most `200 ms` | [x] Met |
| Release regression suite | `32/32` passed | All tests pass | [x] Met |
| Sanitizer regression suite | `31/31` passed | No sanitizer failures | [x] Met |
| Release build warnings | None reported | No compiler warnings | [x] Met |

### Current path priorities

- [x] Repeat the performance measurements on the verified host and record thread count, compiler mode, and build flags.
- [x] Preserve the AVX-512 path while validating the native SIMD and portable fallback paths on non-AVX-512 builds.
- [x] Add repeated-run statistics, p50/p95 latency, variance-relevant samples, and minimum/maximum values to the benchmark output.
- [x] Keep the optional CBLAS path disabled for the target `d=64` workload unless an optimized provider demonstrates a real improvement.
- [x] Add an explicit `--strict` performance-regression gate and register it as an optional CTest.
- [x] Keep numerical, memory-safety, and sanitizer verification mandatory for every optimization change.
- [ ] Repeat the performance measurements on the intended deployment hardware and record CPU model, thread count, compiler, and build flags.
- [ ] Improve the portable non-native fallback until it meets the same latency target, or document a deployment-specific target profile.
- [ ] Do not begin broader architecture expansion until this execution path remains correct and reproducible on the target environment.

## Global Definition of Done

Attention is complete only when all of the following are true:

- [x] The technical specification is approved and versioned.
- [x] The current numerical foundation builds cleanly from an empty build directory and passes the recorded release and sanitizer verification commands.
- [x] The production path has no implicit O(n²) token-pair allocation or computation; the invariant is audited in `docs/COMPLEXITY_AUDIT-2026-08-14.md`.
- [ ] The transformer architecture is implemented, tested, and configurable.
- [ ] The tokenizer is deterministic, versioned, and integrated with all data and inference paths.
- [ ] The dataset pipeline is reproducible, inspected, filtered, split, and traceable.
- [ ] Training is resumable, logged, comparable, and numerically monitored.
- [ ] The model demonstrates measurable language learning on held-out data.
- [ ] The model generates stable, understandable English.
- [ ] The model follows instructions and preserves constraints.
- [ ] The teacher-student protocol works against a mock student.
- [ ] Tool schemas, structured calls, result handling, and error recovery work.
- [ ] External context and memory interfaces work without forcing a fixed memory architecture into the transformer.
- [ ] The model runs locally within defined hardware, memory, latency, and quality limits.
- [ ] The standalone runtime and local API are documented and tested.
- [ ] Regression, safety, reliability, and sustained-runtime tests pass.
- [ ] A clean environment can reproduce the released artifacts and demonstrations.

## Stage 0 — Mission, Scope, and Engineering Baseline

**Purpose:** Freeze the mission, boundaries, measurable requirements, and repository conventions before adding new architecture.

**Prerequisites:** None.

### 0.1 Define the technical mission

- [x] Write the Attention technical specification in the versioned `Goal.md` document.
- [x] Define Attention as a locally runnable transformer and future language-and-teaching foundation.
- [x] Define the intended English capability and record possible Amharic or bilingual support as an explicit design consideration.
- [x] Define the target personal-computer hardware, including CPU, RAM, storage, operating environment, and verification assumptions.
- [x] Define the expected training hardware separately from the expected inference hardware.
- [x] Define the maximum acceptable model size for local use.
- [x] Define the target context length.
- [x] Define acceptable startup time, first-token latency, tokens per second, sustained runtime, memory use, and storage use.
- [x] Define the intended training precision and inference precision.
- [x] Define licensing and data-permission requirements.
- [x] Define which capabilities belong to Attention and which belong to future external systems.
- [x] Define measurable minimum quality for language understanding, generation, instruction following, reasoning demonstrations, and tool interaction.

### 0.2 Establish project conventions

- [x] Define the repository directory structure.
- [x] Define naming conventions for source files, configurations, datasets, checkpoints, experiments, and releases.
- [x] Define the versioning scheme for code, tokenizer, data, model architecture, checkpoints, and APIs.
- [x] Define the experiment-record format.
- [x] Define the required information for a commit or experiment note.
- [x] Define the clean-build command for supported release, sanitizer, and performance configurations.
- [x] Define the test command and minimum test categories.
- [x] Define how failed experiments and known issues are recorded.
- [x] Define how external dependencies are pinned and audited.

### 0.3 Establish the baseline repository state

- [x] Record the current repository commit and branch.
- [x] Record the current source tree and build targets.
- [x] Record the current supported compiler, CMake version, Eigen version, BLAS/LAPACK status, and test framework.
- [x] Record the current clean-build result.
- [x] Record all known compiler warnings.
- [x] Record all known link failures and runtime stubs or intentionally incomplete scope.
- [x] Create the versioned `docs/BASELINE-2026-08-14.md` report before further implementation changes.

**Deliverables:** `Goal.md`, `docs/ENGINEERING_CONVENTIONS.md`, and `docs/BASELINE-2026-08-14.md`.

**Completion gate:** **Complete.** The project mission, constraints, responsibilities, supported baseline, build/test commands, and current limitations are explicit enough that implementation decisions can be evaluated against them.

## Stage 1 — Stabilize the Existing Numerical Foundation

**Purpose:** Make the current SMAO numerical foundation truthful, buildable, testable, and safe before extending it.

**Prerequisites:** Stage 0 complete.

### 1.1 Restore a clean build

- [x] Remove duplicate definitions of `enforce_condition_number_bound`.
- [x] Remove duplicate definitions of `validate_condition_number`.
- [x] Choose one canonical module for shared numerical guards.
- [x] Update headers and translation units so each non-inline function has one definition.
- [x] Run the clean configure step from an empty build directory.
- [x] Build the static library.
- [x] Build the shared library.
- [x] Build the benchmark target.
- [x] Build the complete test executable.
- [x] Run the full test suite.
- [x] Record all remaining compiler warnings, including the warning-free OpenMP-disabled configuration after pragma guards.
- [x] Record the clean-build result in `docs/BASELINE-2026-08-14.md` and `docs/STAGE1_NUMERICAL_GATE_REPORT.md`.

### 1.2 Repair public distance behavior

- [x] Decide the ownership model for `internal_handle`.
- [x] Define the handle structure containing the metric and dimension.
- [x] Populate `internal_handle` during the forward pass.
- [x] Implement `smao_anisotropic_distance` using the stored metric.
- [x] Validate the handle before dereferencing it.
- [x] Validate query and key vector dimensions through the explicit `d` argument.
- [x] Return an error for null or structurally invalid handles; released owning outputs clear their handle.
- [x] Ensure the API never returns success with a fabricated zero distance.
- [x] Release the handle safely in `smao_phase1_release`.
- [x] Add exact-distance, invalid-handle, released-output, dimension-mismatch, and numerical-consistency tests.

### 1.3 Complete Phase 1 gate verification

- [x] Make `check_frozen_gate_criteria` verify output status.
- [x] Verify the minimum eigenvalue threshold explicitly.
- [x] Verify the condition-number threshold explicitly.
- [x] Verify the whitening isometry residual explicitly.
- [x] Verify exact decomposition against the reference dot-product formulation.
- [x] Add an allocation audit confirming that no quadratic auxiliary buffer is allocated.
- [x] Add the throughput and latency benchmark checks.
- [x] Make gate results produce the structured `FrozenGateReport` rather than relying on comments.
- [x] Ensure gate checks are executed in the release verification command and strict CTest performance registration.

### 1.4 Harden the C API

- [x] Check every `malloc` result.
- [x] Release already allocated buffers if a later allocation fails.
- [x] Return `SMAO_ERROR_ALLOCATION` on allocation failure.
- [x] Validate `V` and `d_v` through the internal forward-input contract.
- [x] Validate dimension relationships, including `d_v = d` where required by the current public path.
- [x] Make output initialization safe for repeated calls.
- [x] Make release idempotent.
- [x] Document ownership of every input, output, and handle field.
- [x] Add C-level tests for null pointers, invalid dimensions, NaN, Inf, invalid distance output, and repeated release; allocation-failure cleanup is additionally audited in every allocation branch.

### 1.5 Align implementation and specification

- [x] Document that `epsilon` is validated as a compatibility parameter for decomposition but regularization is applied during metric assembly.
- [x] Reconcile the documented test framework with the actual GoogleTest framework.
- [x] Reconcile declared F16 and BF16 modes with actual implementation support.
- [x] Restrict the public precision contract to supported F32 computation until conversions exist.
- [x] Mark each performance target as target, measured, or verified.
- [x] Mark each numerical claim as tested, audited, or pending.
- [x] Update the README to distinguish the low-level SMAO kernel from the broader Attention system.

**Deliverables:** Clean build, passing tests, functioning dimension-checked distance API, structured Phase 1 gate report, hardened public API, and updated specification status.

**Completion gate:** **Complete.** A clean checkout configures, builds all targets, runs the full release and sanitizer suites, exposes no fabricated public behavior, and produces a complete numerical-gate report.

## Stage 2 — Transformer Architecture Foundation

**Purpose:** Implement the complete transformer computation graph as a modular, testable, configurable system.

**Prerequisites:** Stage 1 complete, or the numerical foundation is explicitly isolated as an independent dependency.

### 2.1 Define architecture configuration

- [x] Define the architecture configuration schema in `include/attention/transformer_config.h`.
- [x] Include vocabulary size, context length, layer count, hidden size, attention-head count, feed-forward size, activation, dropout policy, causal mode, precision, and embedding tying.
- [x] Validate divisibility requirements such as hidden size by head count.
- [x] Calculate parameter count automatically.
- [x] Calculate estimated activation memory automatically.
- [x] Calculate parameter/checkpoint bytes automatically.
- [ ] Calculate full inference-memory estimates once the tensor runtime exists.
- [x] Version the configuration through the source/API contract and tests.
- [ ] Serialize and deserialize configurations deterministically.

### 2.2 Implement tensor and parameter foundations

- [x] Define tensor layout conventions in `docs/TENSOR_PARAMETER_FOUNDATION.md`.
- [x] Define supported data types: F32 only in the first CPU foundation.
- [x] Define CPU device and RAII memory ownership rules.
- [x] Implement parameter initialization.
- [x] Implement deterministic random initialization from a seed.
- [x] Implement gradient storage and gradient clearing.
- [x] Implement checked shape and row-major stride validation.
- [x] Implement checkpoint-compatible stable parameter naming and deterministic lexicographic traversal.
- [x] Add tests for tensor shapes, layouts, initialization, ownership, gradients, and determinism.

### 2.3 Implement the transformer blocks

- [x] Implement token embeddings in `attention::TokenEmbedding`.
- [x] Implement deterministic sinusoidal positional representation in `attention::SinusoidalPositionEncoding`.
- [x] Implement query, key, and value projections in `attention::QKVProjection`.
- [x] Implement streaming causal attention masking in `attention::CausalMask` without a dense sequence-by-sequence tensor.
- [x] Implement the linear feature-map attention score and value aggregation path in `attention::LinearCausalAttention`; dense softmax score matrices remain intentionally excluded.
- [ ] Integrate the intended Attention/SMAO kernel boundary without hiding incompatible semantics.
- [ ] Implement feed-forward or gated feed-forward layers.
- [ ] Implement normalization.
- [ ] Implement residual connections.
- [ ] Implement the final normalization or output processing required by the configuration.
- [ ] Implement vocabulary projection.
- [ ] Implement autoregressive logits output.
- [ ] Make all blocks composable and independently testable.

### 2.4 Implement forward and backward execution

- [ ] Implement a complete forward pass.
- [ ] Implement causal language-model loss.
- [ ] Implement backward propagation.
- [ ] Verify gradient flow through every trainable component.
- [ ] Add numerical gradient checks for selected small configurations.
- [ ] Add deterministic forward tests.
- [ ] Add checkpoint reload equivalence tests.
- [ ] Add malformed-configuration tests.

**Deliverables:** Configurable transformer implementation, parameter and tensor foundation, forward pass, loss, backward pass, architecture report, unit tests.

**Completion gate:** A small transformer passes tensor, mask, forward, loss, gradient, determinism, and checkpoint round-trip tests.

## Current Implementation Position

The current implementation position is **Stage 2.3, with linear aggregation implemented and the Attention/SMAO boundary next**. The verified implementation includes the architecture configuration, F32 CPU tensor foundation, deterministic parameter store, token embeddings, sinusoidal positional representation, per-token QKV projections, a zero-storage streaming causal mask, and `attention::LinearCausalAttention`. The full transformer graph, training execution, and checkpoint serialization are not yet complete.

### Verified State Snapshot

| Area | State | Evidence |
|---|---|---|
| Stage 0 | Complete | `Goal.md`, engineering conventions, baseline report, and all Stage 0 checklist items |
| Stage 1 | Complete | `docs/STAGE1_NUMERICAL_GATE_REPORT.md`; release and sanitizer validation |
| Stage 2.1 | Partially complete | Configuration schema, validation, parameter counts, and activation estimates are complete; full inference-memory estimation and deterministic serialization remain pending |
| Stage 2.2 | Complete | Tensor/parameter contracts and `46/46` foundation verification |
| Stage 2.3 | In progress | Embedding, positional, QKV, causal-mask, and linear aggregation items are complete; SMAO integration and later transformer blocks remain pending |
| Stage 2.4 | Not started | Forward loss, backward propagation, gradient flow, deterministic forward, and checkpoint reload remain pending |
| Current test state | Verified | `56/56` tests pass in Release and portable sanitizer configurations; dedicated linear benchmark also passes |
| Complexity state | Verified | No live-source `n × n`, `sequence_length × sequence_length`, or `context_length × context_length` token-pair allocation pattern |

### Ordered Walkthrough from Todo.md

The next work must follow the checklist in order. The linear attention aggregation contract and implementation are now complete and verified. First, document and test the Attention/SMAO semantic boundary so incompatible metric-aware and feature-map semantics are explicit rather than hidden. Second, implement feed-forward layers, normalization, residual connections, final output processing, vocabulary projection, and autoregressive logits as separate composable modules. Third, complete the Stage 2.1 inference-memory estimate and deterministic configuration serialization. Fourth, implement the Stage 2.4 forward pass, language-model loss, backward propagation, gradient checks, determinism tests, malformed-configuration tests, and checkpoint reload equivalence.

Every next step must update the corresponding checklist item only after its implementation, tests, documentation, and complexity audit have passed. No future attention component may introduce an implicit O(n²) token-pair computation or allocation.

## Stage 3 — Tokenizer and Text Representation

**Purpose:** Create the stable text representation shared by training, evaluation, inference, conversations, and tools.

**Prerequisites:** Stage 0 language scope defined; Stage 2 model vocabulary interface defined.

### 3.1 Select and implement the tokenizer

- [ ] Choose the tokenizer algorithm based on language and domain requirements.
- [ ] Define normalization for whitespace.
- [ ] Define normalization for punctuation.
- [ ] Define capitalization behavior.
- [ ] Define number and symbol behavior.
- [ ] Define code-text behavior.
- [ ] Define treatment of malformed Unicode.
- [ ] Define beginning-of-sequence token.
- [ ] Define end-of-sequence token.
- [ ] Define padding token.
- [ ] Define unknown token behavior.
- [ ] Define conversation-role tokens.
- [ ] Define tool-call and tool-result tokens.
- [ ] Define memory-context markers.
- [ ] Define lesson and feedback markers for the teacher interface.

### 3.2 Validate tokenization

- [ ] Test deterministic encoding.
- [ ] Test deterministic decoding.
- [ ] Test encode-decode round trips.
- [ ] Measure token efficiency on English.
- [ ] Measure token efficiency on Amharic.
- [ ] Measure token efficiency on bilingual text.
- [ ] Measure token efficiency on code.
- [ ] Measure token efficiency on structured API messages.
- [ ] Measure unknown-token rate.
- [ ] Measure sequence-length distribution.
- [ ] Freeze the tokenizer version for each training run.
- [ ] Store vocabulary and configuration with every checkpoint.

**Deliverables:** Versioned tokenizer, special-token specification, tokenization report, tokenizer tests.

**Completion gate:** The same tokenizer artifacts are used consistently by dataset preparation, training, evaluation, and inference, and all deterministic round-trip tests pass.

## Stage 4 — Dataset and Data-Lineage Pipeline

**Purpose:** Construct a clean, permitted, inspectable, and reproducible corpus appropriate for the intended model behavior.

**Prerequisites:** Stage 0 language and domain scope; Stage 3 tokenizer stable enough for statistics.

### 4.1 Define data policy

- [ ] Define permitted data sources.
- [ ] Record source ownership and license information.
- [ ] Define excluded sources and prohibited data.
- [ ] Define sensitive-data handling and removal rules.
- [ ] Define language proportions.
- [ ] Define domain proportions.
- [ ] Define code and structured-text proportions.
- [ ] Define data retention and deletion rules.
- [ ] Define dataset versioning and manifest format.

### 4.2 Implement ingestion and normalization

- [ ] Implement source ingestion.
- [ ] Normalize documents into an internal format.
- [ ] Normalize encoding and line endings.
- [ ] Detect language.
- [ ] Preserve source metadata.
- [ ] Record source URL or identifier where permitted.
- [ ] Remove empty and corrupted documents.
- [ ] Filter spam-like and extremely low-quality content.
- [ ] Filter unsafe or disallowed content according to the data policy.
- [ ] Normalize document boundaries.
- [ ] Generate inspectable samples.

### 4.3 Implement quality and leakage controls

- [ ] Remove exact duplicates.
- [ ] Remove near-duplicates.
- [ ] Detect repeated boilerplate.
- [ ] Split training, validation, and test sets without leakage.
- [ ] Verify that evaluation examples are not present in training data.
- [ ] Record filtering counts and reasons.
- [ ] Record language distribution.
- [ ] Record domain distribution.
- [ ] Record document count.
- [ ] Record token count.
- [ ] Record sequence-length distribution.
- [ ] Record duplication rate.
- [ ] Record rejected-document statistics.

### 4.4 Build tokenized shards

- [ ] Tokenize the cleaned corpus.
- [ ] Pack sequences according to context length.
- [ ] Write versioned training shards.
- [ ] Write validation shards.
- [ ] Write test shards.
- [ ] Store tokenizer version in the manifest.
- [ ] Store source and filtering metadata in the manifest.
- [ ] Verify shard checksums.
- [ ] Add a data-loader inspection command.

**Deliverables:** Data policy, source manifest, reproducible cleaning pipeline, dataset statistics, tokenized shards, lineage manifest.

**Completion gate:** A dataset can be rebuilt from documented inputs and the project can identify exactly which data version produced every training run and checkpoint.

## Stage 5 — Reproducible Training System

**Purpose:** Turn model training into a controlled, resumable, inspectable, and comparable engineering process.

**Prerequisites:** Stages 2–4 complete.

### 5.1 Implement the training loop

- [ ] Implement batch loading.
- [ ] Implement sequence packing.
- [ ] Implement causal language-model loss.
- [ ] Implement forward execution in training mode.
- [ ] Implement backward propagation.
- [ ] Implement optimizer integration.
- [ ] Implement learning-rate scheduling.
- [ ] Implement gradient accumulation.
- [ ] Implement gradient clipping where required.
- [ ] Implement validation evaluation.
- [ ] Implement configurable logging intervals.

### 5.2 Implement run management

- [ ] Define the training-run configuration format.
- [ ] Record random seed.
- [ ] Record code commit.
- [ ] Record architecture configuration.
- [ ] Record tokenizer version.
- [ ] Record dataset version.
- [ ] Record hardware information.
- [ ] Record compiler and dependency versions.
- [ ] Record training duration.
- [ ] Record tokens processed.
- [ ] Record loss and validation loss.
- [ ] Record learning rate.
- [ ] Record throughput.
- [ ] Record memory use.
- [ ] Record gradient norms.
- [ ] Record warnings and failures.

### 5.3 Implement checkpointing and recovery

- [ ] Save model parameters.
- [ ] Save optimizer state.
- [ ] Save scheduler state.
- [ ] Save gradient-accumulation state where required.
- [ ] Save global step and token count.
- [ ] Save configuration and metadata.
- [ ] Save tokenizer version.
- [ ] Save dataset version.
- [ ] Save checksums.
- [ ] Resume after interruption.
- [ ] Test resume equivalence against uninterrupted training.
- [ ] Retain best validation checkpoint.
- [ ] Retain periodic checkpoints.

### 5.4 Add numerical diagnostics

- [ ] Detect NaNs.
- [ ] Detect Infs.
- [ ] Detect exploding gradients.
- [ ] Detect stalled loss.
- [ ] Detect invalid learning-rate values.
- [ ] Detect data-loader failures.
- [ ] Detect memory exhaustion.
- [ ] Add short diagnostic runs before long runs.
- [ ] Add a tiny overfit configuration.
- [ ] Store diagnostic reports with each run.

**Deliverables:** Training executable, run configuration, logging system, checkpoint system, resume support, diagnostics, experiment reports.

**Completion gate:** A run can stop and resume, produces complete metadata, and can be compared against another run without ambiguity.

## Stage 6 — Basic Language-Learning Proof

**Purpose:** Demonstrate that the architecture and training pipeline learn meaningful language behavior before adding advanced adaptation.

**Prerequisites:** Stage 5 complete.

- [ ] Run the tiny-data overfitting test.
- [ ] Verify that loss decreases on the tiny dataset.
- [ ] Verify that generated samples memorize the tiny diagnostic set as expected.
- [ ] Run a small held-out training experiment.
- [ ] Compare against an untrained baseline.
- [ ] Measure training loss.
- [ ] Measure validation loss.
- [ ] Compare generations across checkpoints.
- [ ] Test spelling and word boundaries.
- [ ] Test punctuation.
- [ ] Test grammar patterns.
- [ ] Test common structural patterns.
- [ ] Check for repetition loops.
- [ ] Check for broken syntax.
- [ ] Check for data leakage.
- [ ] Check for overfitting.
- [ ] Record failure patterns and hypotheses.
- [ ] Preserve failed experiments in the experiment log.

**Deliverables:** Learning-proof report, baseline comparison, checkpoint samples, failure analysis.

**Completion gate:** Attention improves on held-out data relative to an untrained baseline and produces coherent text at the selected scale.

## Stage 7 — English Capability and Evaluation

**Purpose:** Establish stable, understandable English behavior across the intended task categories.

**Prerequisites:** Stage 6 complete.

### 7.1 Create the fixed evaluation set

- [ ] Define evaluation categories.
- [ ] Create grammar examples.
- [ ] Create syntax examples.
- [ ] Create vocabulary examples.
- [ ] Create coherence examples.
- [ ] Create short-generation examples.
- [ ] Create long-generation examples.
- [ ] Create explanation examples.
- [ ] Create summary examples.
- [ ] Create structured-writing examples.
- [ ] Create dialogue examples.
- [ ] Create multi-turn continuity examples.
- [ ] Create factual-consistency probes.
- [ ] Exclude the evaluation set from training.
- [ ] Version and checksum the evaluation set.

### 7.2 Measure language behavior

- [ ] Evaluate grammar.
- [ ] Evaluate syntax.
- [ ] Evaluate vocabulary.
- [ ] Evaluate coherence.
- [ ] Evaluate instruction comprehension.
- [ ] Evaluate context-length effects.
- [ ] Evaluate decoding-parameter effects.
- [ ] Record repetition failures.
- [ ] Record irrelevant continuations.
- [ ] Record unsupported claims.
- [ ] Record broken syntax.
- [ ] Use human review for samples that metrics cannot judge.
- [ ] Compare every release candidate with the fixed baseline.

**Deliverables:** Versioned English evaluation set, evaluation runner, language-quality report, failure catalogue.

**Completion gate:** Attention communicates in understandable and stable English across all defined evaluation categories.

## Stage 8 — Instruction Following and Conversation Format

**Purpose:** Convert the base language model into a model that listens, follows constraints, and responds in a useful structured manner.

**Prerequisites:** Stage 7 complete; tokenizer supports conversation markers.

### 8.1 Define the instruction protocol

- [ ] Define system message format.
- [ ] Define user message format.
- [ ] Define assistant message format.
- [ ] Define context representation.
- [ ] Define tool-result representation.
- [ ] Define stop conditions.
- [ ] Define malformed-message behavior.
- [ ] Define conversation serialization.
- [ ] Version the protocol.

### 8.2 Create instruction data

- [ ] Create ordinary instruction examples.
- [ ] Create constrained instruction examples.
- [ ] Create clarification-required examples.
- [ ] Create limitation and uncertainty examples.
- [ ] Create style-controlled examples.
- [ ] Create structured-output examples.
- [ ] Create conflicting-instruction examples.
- [ ] Create malformed-input examples.
- [ ] Review examples for quality.
- [ ] Split instruction data into train, validation, and test sets.

### 8.3 Fine-tune and validate

- [ ] Implement supervised fine-tuning.
- [ ] Preserve base language capability.
- [ ] Evaluate instruction adherence.
- [ ] Evaluate constraint preservation.
- [ ] Evaluate clarification behavior.
- [ ] Evaluate limitation statements.
- [ ] Evaluate requested response styles.
- [ ] Evaluate resistance to conflicting instructions.
- [ ] Compare the fine-tuned model with the base model.
- [ ] Record regressions and improvements.

**Deliverables:** Conversation protocol, instruction dataset, fine-tuning configuration, instruction-following checkpoint, evaluation report.

**Completion gate:** Attention follows ordinary instructions reliably while preserving constraints and asking for clarification when required.

## Stage 9 — Teacher-Student Interface

**Purpose:** Make Attention usable as a structured teacher for an external student or cognitive system without coupling the transformer to a fixed future architecture.

**Prerequisites:** Stage 8 complete.

### 9.1 Define protocol messages

- [ ] Define observation messages.
- [ ] Define question messages.
- [ ] Define attempted-action messages.
- [ ] Define tool-result messages.
- [ ] Define internal-state messages where permitted.
- [ ] Define explanation messages.
- [ ] Define demonstration messages.
- [ ] Define correction messages.
- [ ] Define plan messages.
- [ ] Define approval messages.
- [ ] Define rejection messages.
- [ ] Define uncertainty messages.
- [ ] Define lesson-completion messages.
- [ ] Define protocol versioning.

### 9.2 Implement teacher responses

- [ ] Implement structured teaching responses.
- [ ] Implement explanations.
- [ ] Implement demonstrations.
- [ ] Implement corrections.
- [ ] Implement plans.
- [ ] Implement confidence or uncertainty reporting.
- [ ] Keep language generation separate from approval and reward mechanisms.
- [ ] Keep environment control outside the model.
- [ ] Log teacher-student interactions.
- [ ] Make interaction logs replayable.

### 9.3 Build the mock student

- [ ] Implement a minimal mock student.
- [ ] Send observations to Attention.
- [ ] Send questions to Attention.
- [ ] Send attempted actions to Attention.
- [ ] Receive structured guidance.
- [ ] Apply a correction.
- [ ] Repeat a task after teaching.
- [ ] Measure whether the mock student improves.
- [ ] Record unsuccessful teaching interactions.

**Deliverables:** Teacher-student protocol, schemas, mock student, interaction logger, teacher evaluation report.

**Completion gate:** A mock student can communicate with Attention and receive useful, structured teaching responses through the documented protocol.

## Stage 10 — Tools, APIs, Context, and Memory Interfaces

**Purpose:** Prepare Attention to operate as a controlled component inside larger systems.

**Prerequisites:** Stage 9 complete; structured message format stable.

### 10.1 Tool and API schemas

- [ ] Define machine-readable tool schemas.
- [ ] Define tool names and descriptions.
- [ ] Define required fields.
- [ ] Define optional fields.
- [ ] Define field types.
- [ ] Define validation rules.
- [ ] Define error messages.
- [ ] Define confirmation requirements.
- [ ] Define permission levels.
- [ ] Define tool-call serialization.
- [ ] Define tool-result serialization.
- [ ] Version the tool protocol.

### 10.2 Implement tool interaction

- [ ] Teach the model to distinguish text from tool calls.
- [ ] Generate parseable tool calls.
- [ ] Validate tool calls before execution.
- [ ] Insert tool results into context.
- [ ] Continue a task after a successful result.
- [ ] Recover from invalid calls.
- [ ] Recover from tool failures.
- [ ] Support multiple sequential calls.
- [ ] Require confirmation for dangerous or irreversible actions.
- [ ] Log every request, call, result, error, and follow-up.
- [ ] Test mock tools before connecting real tools.

### 10.3 Context and memory interface

- [ ] Define recent-conversation context.
- [ ] Define retrieved-memory context.
- [ ] Define external-observation context.
- [ ] Define source metadata.
- [ ] Define timestamps.
- [ ] Define confidence.
- [ ] Define relevance.
- [ ] Mark current context separately from recalled information.
- [ ] Test incomplete context.
- [ ] Test contradictory context.
- [ ] Test stale context.
- [ ] Test overlong context.
- [ ] Implement truncation or prioritization.
- [ ] Keep memory storage external and replaceable.

**Deliverables:** Tool schema, tool runtime adapter, mock-tool suite, context schema, memory interface, interaction logs.

**Completion gate:** Attention can make valid mock-tool calls, use results, recover from errors, and consume structured external context without embedding a premature memory architecture.

## Stage 11 — Efficiency and Local Inference Runtime

**Purpose:** Make the trained model practical for sustained local use on the defined personal-computer target.

**Prerequisites:** Stage 8 complete for basic runtime work; Stage 10 complete for structured runtime work.

### 11.1 Establish performance baselines

- [ ] Measure parameter count.
- [ ] Measure checkpoint size.
- [ ] Measure memory footprint.
- [ ] Measure startup time.
- [ ] Measure first-token latency.
- [x] Measure tokens per second.
- [x] Measure sustained throughput.
- [ ] Measure CPU and GPU utilization.
- [ ] Measure memory bandwidth where relevant.
- [ ] Measure energy or thermal behavior where available.
- [x] Record quality and numerical validity alongside every performance measurement.

### 11.2 Optimize inference

- [x] Implement and verify the intended F32 local precision path.
- [ ] Investigate reduced precision.
- [ ] Investigate quantization.
- [ ] Investigate weight sharing.
- [ ] Investigate pruning.
- [ ] Implement key-value caching where applicable.
- [x] Reduce unnecessary memory copies through fused kernels and reusable output workspaces.
- [x] Reduce startup and steady-state overhead in the Phase 1 forward path.
- [x] Compare numerical validity and regression results before and after every completed optimization.
- [ ] Reject optimizations that violate quality thresholds.
- [ ] Package the model without requiring the full training environment.

### 11.3 Build the runtime

- [ ] Implement checkpoint validation.
- [ ] Implement checkpoint loading.
- [ ] Implement text generation.
- [ ] Implement structured-message generation.
- [ ] Implement temperature or equivalent decoding controls.
- [ ] Implement top-p or equivalent sampling controls.
- [ ] Implement maximum output length.
- [ ] Implement stop conditions.
- [ ] Implement streaming output where useful.
- [ ] Implement request cancellation.
- [ ] Implement timeout handling.
- [ ] Implement clear configuration errors.
- [ ] Implement clear context-length errors.
- [ ] Implement clear memory errors.
- [ ] Implement deterministic inference mode.
- [ ] Implement a command-line interface.
- [ ] Implement a local API.
- [ ] Document all runtime endpoints and command-line options.

**Deliverables:** Performance baseline, optimized checkpoint, standalone runtime, CLI, local API, runtime documentation.

**Completion gate:** The optimized model runs on target personal hardware for a sustained session within the defined memory, latency, throughput, and quality thresholds.

## Stage 12 — Evaluation, Regression, Reliability, and Safety

**Purpose:** Establish confidence that every release is correct, measurable, safe to integrate, and resistant to regressions.

**Prerequisites:** Stages 1–11 progressively complete.

### 12.1 Automated tests

- [ ] Test tokenizer determinism.
- [ ] Test tokenizer round trips.
- [ ] Test tensor shapes.
- [ ] Test causal masking.
- [ ] Test attention computation.
- [ ] Test feed-forward computation.
- [ ] Test normalization.
- [ ] Test residual connections.
- [ ] Test loss calculation.
- [ ] Test gradients.
- [ ] Test checkpoint save and load.
- [ ] Test training resumption.
- [ ] Test inference determinism.
- [ ] Test tool-call parsing.
- [ ] Test tool-result handling.
- [ ] Test context serialization.
- [ ] Test runtime errors.
- [ ] Test cancellation and timeouts.

### 12.2 Regression evaluation

- [ ] Create a fixed baseline checkpoint.
- [ ] Compare language quality against baseline.
- [ ] Compare instruction adherence against baseline.
- [ ] Compare tool-call validity against baseline.
- [ ] Compare context behavior against baseline.
- [ ] Compare memory use against baseline.
- [ ] Compare latency against baseline.
- [ ] Compare throughput against baseline.
- [ ] Compare sustained-runtime stability against baseline.
- [ ] Require a written explanation for every critical regression.
- [ ] Store an evaluation report with every release candidate.

### 12.3 Reliability and safety controls

- [ ] Keep generated text separate from executed actions.
- [ ] Validate every tool call externally.
- [ ] Enforce filesystem permissions.
- [ ] Enforce network permissions.
- [ ] Enforce operating-system permissions.
- [ ] Enforce code-execution permissions.
- [ ] Log model decisions and tool interactions.
- [ ] Define uncertainty behavior.
- [ ] Define refusal behavior.
- [ ] Test malformed inputs.
- [ ] Test prompt injection.
- [ ] Test conflicting context.
- [ ] Test malformed tool results.
- [ ] Treat generated code as untrusted.
- [ ] Add review or sandboxing before executing generated code.
- [ ] Add an emergency stop mechanism.
- [ ] Verify that privileged actions require explicit external authorization.

**Deliverables:** Automated test suite, regression reports, reliability report, safety test suite, release-candidate checklist.

**Completion gate:** Every release candidate has a complete evaluation report, no unresolved critical regression, and no path from ordinary generated text directly to privileged action.

## Stage 13 — Documentation, Packaging, and Reproducible Release

**Purpose:** Make the system understandable, rebuildable, runnable, and integrable by the future project maintainer or another engineer.

**Prerequisites:** All required implementation stages and gates complete.

### 13.1 Documentation

- [ ] Document the architecture.
- [ ] Document the mathematical operations.
- [ ] Document the tokenizer.
- [ ] Document the data policy.
- [ ] Document data construction and limitations.
- [ ] Document training configurations.
- [ ] Document checkpoint metadata.
- [ ] Document capabilities.
- [ ] Document known failure modes.
- [ ] Document installation.
- [ ] Document local execution.
- [ ] Document CLI usage.
- [ ] Document local API usage.
- [ ] Document teacher-student protocol.
- [ ] Document tool schemas.
- [ ] Document context and memory interfaces.
- [ ] Document evaluation procedures.
- [ ] Document safety boundaries.
- [ ] Add a changelog.
- [ ] Preserve failed experiments and lessons learned.

### 13.2 Release artifacts

- [ ] Package source code.
- [ ] Package configuration files.
- [ ] Package tokenizer artifacts.
- [ ] Package dataset manifests.
- [ ] Package model checkpoints.
- [ ] Package runtime binaries or build instructions.
- [ ] Package evaluation scripts.
- [ ] Package test commands.
- [ ] Generate checksums.
- [ ] Record compiler and dependency versions.
- [ ] Record hardware used for verification.
- [ ] Record known limitations.
- [ ] Assign a release version.

### 13.3 Clean-environment reproduction

- [ ] Create a clean environment.
- [ ] Install dependencies from documented instructions.
- [ ] Build from source.
- [ ] Run unit tests.
- [ ] Run integration tests.
- [ ] Load the released checkpoint.
- [ ] Run the evaluation suite.
- [ ] Start the local runtime.
- [ ] Run documented CLI demonstrations.
- [ ] Run documented API demonstrations.
- [ ] Verify tool-call demonstrations with mock tools.
- [ ] Verify teacher-protocol demonstrations with the mock student.
- [ ] Compare results with the release report.

**Deliverables:** Complete documentation, release package, checksums, reproducibility report, changelog.

**Completion gate:** A clean environment can install, build, test, evaluate, run, and reproduce the documented Attention demonstrations.

## Dependency Order

| Stage | Depends on | Must produce |
|---|---|---|
| 0. Mission and baseline | None | Approved specification and baseline |
| 1. Numerical foundation | Stage 0 | Clean, truthful, tested current foundation |
| 2. Transformer architecture | Stage 1 or isolated kernel boundary | Tested transformer computation graph |
| 3. Tokenizer | Stage 0 and model vocabulary interface | Versioned tokenizer |
| 4. Data pipeline | Stage 0 and tokenizer | Reproducible dataset artifacts |
| 5. Training | Stages 2–4 | Resumable training system |
| 6. Learning proof | Stage 5 | Evidence of language learning |
| 7. English capability | Stage 6 | Fixed evaluation and language report |
| 8. Instruction following | Stage 7 | Instruction-tuned checkpoint |
| 9. Teacher interface | Stage 8 | Working teacher-student protocol |
| 10. Tools and context | Stage 9 | Tool and external-context integration |
| 11. Local runtime | Stage 8; Stage 10 for structured operation | Efficient runtime and API |
| 12. Evaluation and safety | Progressive completion of Stages 1–11 | Release-candidate evidence |
| 13. Release | Stage 12 | Reproducible Attention release |

## Immediate Execution Queue

These are the first tasks to execute against the current repository before beginning broader model work:

- [ ] Create or restore the canonical `Goal.md` source document if it is intended to be part of the repository.
- [ ] Record the current repository commit and clean-build baseline.
- [ ] Remove the duplicate numerical-guard definitions.
- [ ] Run a clean CMake configure.
- [ ] Build static and shared libraries.
- [ ] Build the complete test target.
- [ ] Run the complete test suite.
- [ ] Record all remaining compiler warnings.
- [ ] Decide the public distance-handle ownership model.
- [ ] Replace the public distance stub with a correct implementation or remove it temporarily.
- [ ] Add allocation-failure and handle-lifecycle tests.
- [ ] Complete the frozen-gate checks.
- [ ] Reconcile declared precision modes with actual behavior.
- [ ] Update the repository README to separate verified behavior from target behavior.
- [ ] Commit the stabilized numerical foundation before starting new architecture.

## Release-Blocking Conditions

The following conditions block release and block claiming completion of the relevant stage:

- [ ] Any clean-build linker failure remains.
- [ ] Any public API returns success for a fabricated or uncomputed result.
- [ ] Any checkpoint cannot be loaded and validated.
- [ ] Any training run cannot resume correctly.
- [ ] Any dataset cannot be traced to a versioned manifest.
- [ ] Any evaluation set is present in the training data.
- [ ] Any critical regression is unexplained.
- [ ] Any privileged action can be triggered directly by unvalidated model text.
- [ ] Any documented capability is not distinguished from a target or unverified claim.
- [ ] A clean environment cannot reproduce the release.

## Change Record

| Date | Change | Evidence | Status |
|---|---|---|---|
| 2026-08-13 | Converted the Attention objective and staged flow into this implementation checklist | `Attention_Objective.md`, `Phase-1.md`, repository inspection | Complete |
| 2026-08-13 | Recorded missing `Goal.md` source document | Workspace search | Action required |

## References

[1]: https://github.com/nexuss0781/Attention "Attention repository"

[1] [Attention repository](https://github.com/nexuss0781/Attention)

*Private roadmap terminology intentionally omitted.*
*Document title: Attention Todo*
*Author: Manus AI*
