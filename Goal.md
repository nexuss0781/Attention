# Attention Mission and Technical Specification

**Document version:** 0.1.0

**Status:** Stage 0 baseline specification

**Date:** 2026-08-14

**Repository:** `nexuss0781/Attention`

## Mission

Attention is a locally runnable, efficient, and inspectable intelligence foundation. Its immediate engineering responsibility is to provide a correct numerical kernel and a configurable transformer foundation that can later support language learning, instruction following, teaching interactions, tools, external context, and a larger cognitive system.

The project must prioritize **correctness, local operability, reproducibility, measurable performance, memory safety, and modular integration**. No capability may be represented as complete until it is implemented, tested, benchmarked where applicable, and documented.

## Scope

The current scope includes the existing SMAO numerical foundation and the first transformer-architecture foundation. The numerical foundation contains metric assembly, conditioning, whitening, decomposition, anisotropic distance, numerical guards, the C and C++ APIs, benchmarks, and regression tests. The transformer foundation currently contains validated architecture configuration and resource estimates; transformer tensors, parameters, computation blocks, training, and text generation remain future implementation work.

The planned system will eventually include a causal transformer, tokenizer, reproducible dataset pipeline, training and checkpoint recovery, English language evaluation, instruction following, teacher-student interaction, tool and API protocols, external context and memory interfaces, and a standalone local runtime.

The project does not claim that the current repository is an AGI, ASI, complete language model, autonomous operating system, humanoid system, or finished cognitive organism. Those are outside the current implementation scope.

## Language and Domain Scope

English is the first required language for model communication and evaluation. Amharic and bilingual support are explicit future design considerations and must be measured at the tokenizer and dataset stages before being treated as supported capabilities.

The training corpus must be intentionally selected, permitted for use, traceable, filtered, deduplicated, and split without evaluation leakage. Sources must have a documented license or permission basis, and prohibited or unverified sources must be excluded. The project must not assume that indiscriminate ingestion of the entire internet is necessary or acceptable.

## Target Hardware Profiles

### Development and verification profile

| Property | Stage 0 recorded value |
|---|---|
| Architecture | x86_64 |
| CPU | Intel(R) Xeon(R) Processor @ 2.50GHz |
| Visible logical CPUs | 6 |
| Threads per core | 2 |
| Memory | 3.8 GiB total, approximately 2.9 GiB available at capture |
| Swap | 2.0 GiB |
| Root storage | 40 GiB total, approximately 29 GiB available at capture |
| Operating environment | Ubuntu 24.04 sandbox |

This environment is a verification profile, not a guarantee of the user’s eventual deployment hardware. Performance claims must record CPU model, visible threads, compiler, build type, SIMD mode, OpenMP thread count, and benchmark repetition count.

### Inference profile

The intended inference profile is a personal computer capable of running the selected model locally without requiring a remote service. The initial local-model budget is a checkpoint of at most **1.5 GiB** and a steady-state working set of at most **3.0 GiB** on the recorded verification environment. The initial context-length target is **2,048 tokens**. The initial runtime targets are startup within **5 seconds**, first-token latency within **500 milliseconds** after warm-up, and a separately measured sustained generation rate that must be recorded before it is claimed.

The final model size, context length, memory budget, startup time, first-token latency, sustained throughput, and precision must be selected together rather than independently.

The current verified numerical target profile is `n=1,000,000`, `d=64` for the Phase 1 kernel. The native optimized path has a p95 decomposition target of at least 5 million tokens per second and a p95 end-to-end target of at most 200 milliseconds. The portable path must remain numerically correct even when its performance differs from the native target profile.

### Training profile

Training hardware is intentionally separate from inference hardware. Training configurations must record available CPU/GPU devices, memory, precision, batch size, gradient accumulation, sequence length, optimizer, scheduler, and checkpoint storage requirements before a training run begins.

## Model and Runtime Constraints

The first transformer foundation is causal and F32-only until other precisions have real conversion, computation, and test paths. The architecture must validate vocabulary size, context length, layer count, hidden size, attention-head count, feed-forward size, activation, dropout, causal mode, and embedding-tying behavior.

The local runtime must eventually expose deterministic and configurable generation, streaming where supported, cancellation, timeouts, clear error states, checkpoint validation, and a documented local API. Generated text must remain separate from executed actions, and privileged actions must require external validation and authorization.

## Quality Requirements

Every implemented component must have a clearly stated contract and corresponding tests. The minimum quality requirements are:

| Area | Minimum requirement |
|---|---|
| Numerical correctness | Finite-input handling, SPD and conditioning checks, whitening consistency, decomposition identity, and explicit failure states |
| Memory safety | Clean AddressSanitizer and UndefinedBehaviorSanitizer runs for supported test paths |
| Build quality | Clean configure and build from an empty directory with no unresolved warnings or link failures |
| API reliability | Valid ownership, allocation failure handling, idempotent release, dimension checks, and no fabricated success values |
| Language capability | Later stages must measure grammar, syntax, vocabulary, coherence, dialogue, summaries, explanations, and context behavior |
| Instruction following | Later stages must measure constraint preservation, clarification, uncertainty, and conflicting instructions |
| Tool interaction | Later stages must validate schema-conformant calls, result handling, permission boundaries, and error recovery |
| Performance | Benchmarks must report repeated samples and p95 values; strict targets must be enforced only on applicable hardware profiles |
| Reproducibility | Source, configuration, data, tokenizer, checkpoint, compiler, dependency, hardware, and test metadata must be recorded |

## Minimum Future Model Quality Gates

These are initial acceptance thresholds for later model stages and are not claims about the current repository:

- The base model must reduce validation loss relative to an untrained baseline on a held-out corpus.
- The model must produce syntactically understandable English on the fixed evaluation set.
- The instruction model must preserve required constraints on the fixed instruction set and ask for clarification on underspecified tasks.
- The tool interface must produce schema-valid calls and recover from invalid tool results in the mock-tool suite.
- Every quality claim must include the model version, tokenizer version, data version, evaluation version, sampling settings, and reproducible command.

## Responsibility Boundaries

Attention itself is responsible for the numerical kernel, transformer architecture, tokenizer integration, training interfaces, inference runtime, structured protocols, evaluation, and documentation that are implemented in this repository.

External systems may provide operating-system services, storage, authentication, scheduling, memory retrieval, tool execution, hardware drivers, or future embodiment. Attention must communicate with such systems through explicit interfaces and must not silently assume that an external capability exists.

## Versioning and Release Policy

The repository uses semantic project versions for releases and Git commit history for implementation traceability. Architecture configurations, tokenizer artifacts, datasets, checkpoints, APIs, and evaluation suites must each carry an explicit version or content identifier.

A release requires a clean build, passing tests, documented benchmark results, known limitations, reproducible instructions, and a record of the exact source commit and configuration used for verification.

## Stage 0 Acceptance

Stage 0 is complete when this specification, the engineering conventions, and the baseline report are committed; the repository state is recorded; the clean build and test commands are reproducible; and every later implementation decision can be evaluated against this document.
