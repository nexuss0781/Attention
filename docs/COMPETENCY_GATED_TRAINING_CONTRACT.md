# Competency-Gated English Continual-Training Contract

## Purpose

Attention training will proceed as a sequence of bounded English learning sessions. Each session trains from exactly one parent checkpoint on one deterministic document chunk, evaluates explicit competencies, saves an immutable session artifact, and stops for manual review. A session may produce a child checkpoint only after its automated gates pass and the user records a learning decision.

The workflow is intentionally **continual and inspectable**, not an uncontrolled long-running training job. Every session must be independently reproducible from its parent checkpoint, dataset revision, document range, tokenizer metadata, architecture configuration, random seed, optimizer configuration, and source checksums.

## Session lifecycle

| State | Meaning | Allowed transition |
|---|---|---|
| `PLANNED` | Session manifest is complete but no training has started | `RUNNING` |
| `RUNNING` | Training is executing on the selected chunk | `VALIDATING` or `FAILED` |
| `VALIDATING` | Automated loss, competency, integrity, and regression checks are executing | `AWAITING_REVIEW` or `FAILED` |
| `AWAITING_REVIEW` | Artifacts are complete and the process has stopped for the user's manual learning judgment | `PROMOTED`, `RETRY`, or `ABORTED` |
| `PROMOTED` | The session checkpoint is accepted as the next parent checkpoint | `PLANNED` for a later session |
| `RETRY` | The competency was not accepted; the next session must use an explicitly chosen retry chunk or configuration | `PLANNED` |
| `FAILED` | A deterministic or integrity gate failed before manual review | `RETRY` or `ABORTED` |
| `ABORTED` | The branch is stopped and cannot be used as a parent without an explicit override | terminal |

`run.sh` may automate transitions through `VALIDATING`, but it must stop at `AWAITING_REVIEW`. It must never silently promote a checkpoint or silently continue to a new competency.

## What a session contains

A session manifest must identify the parent checkpoint, one approved English dataset release, one deterministic document range or shard list, the training objective, the competency being attempted, the validation examples, the random seed, optimizer settings, and output paths. The selected chunk must be bounded by document count and token count, and its exact ordered document IDs must be recorded after filtering and deduplication.

The session artifact directory is immutable after finalization and contains the input manifest, resolved manifest, source and chunk checksums, tokenizer metadata, architecture serialization, dependency versions, training log, validation report, competency report, model checkpoint, training-state checkpoint, and manual decision record. A separate small pointer may identify the currently promoted checkpoint; historical session directories must never be overwritten.

## Chunk selection

A chunk is selected from an approved source pool using a deterministic seed and a stable ordering key. The selector records the source revision, shard or document range, selected document IDs, raw and normalized token counts, and the SHA-256 digest of the packed token stream. By default, a later session must not reuse a prior training document. Reuse is allowed only when the competency manifest explicitly declares a repeat or remediation session.

Training and validation documents must be disjoint by document identity before token packing. The validation set must not be sampled from the same normalized document as the training chunk unless the competency is explicitly a within-document diagnostic and the report labels it as such.

## Competency gates

Each competency has four layers. The first is a deterministic integrity gate: all inputs, outputs, losses, gradients, and parameters are finite; the tokenizer and architecture match the parent; the checkpoint reloads; and the no-quadratic-computation scan remains clean. The second is a numerical learning gate: training loss must improve from its baseline on the selected chunk, and validation loss must be finite and recorded. The third is a competency evaluation gate: a fixed, held-out set of examples tests the intended skill with explicit scoring rules. The fourth is manual review: the user inspects representative outputs or reports an external test result and chooses `PROMOTE`, `RETRY`, or `ABORT`.

A failure on the same competency across multiple independent chunks is evidence for investigation, not automatic proof of an architectural defect. The diagnosis must first separate data quality, label/target construction, tokenizer coverage, optimization settings, and evaluation errors from model-capacity or architectural limitations.

## Checkpoint lineage

Every session after the initial checkpoint must load a fresh model and empty parameter store from the exact parent checkpoint. The session checkpoint stores the updated model parameters and the training-state envelope stores optimizer and progress state. Promotion records the child checkpoint digest, parent digest, session ID, competency result, and manual decision. A failed or unreviewed checkpoint cannot become the parent of a normal next session.

## Required manual decision file

The process stops with a template containing the session ID, competency, automated results, links to outputs, and three explicit choices:

```text
DECISION: PROMOTE | RETRY | ABORT
COMPETENCY: <competency_id>
RATIONALE: <human assessment>
NEXT_ACTION: <next competency or retry plan>
```

The next invocation of `run.sh` reads this decision and refuses to continue when it is absent, malformed, or inconsistent with the session state.

## Initial English competency ladder

The first ladder should progress from basic causal language modeling to increasingly useful language behavior:

1. **Causal basics:** stable tokenization, next-token loss decrease, checkpoint reload, and deterministic continuation.
2. **English syntax and local coherence:** agreement, punctuation, sentence completion, and short paragraph continuation.
3. **Factual and explanatory language:** concise explanations, definitions, and evidence-preserving summaries.
4. **Instruction following:** obeying explicit constraints, formatting requests, and multi-step directions.
5. **Ambiguity handling:** recognizing materially ambiguous requests, asking one useful clarification, and avoiding an unjustified assumption.
6. **Conversational quality:** helpfulness, correctness, coherence, calibrated uncertainty, and concise answers across multi-turn English conversations.

A later competency may not be promoted merely because its training loss decreases. It requires its own held-out competency report and manual review.
