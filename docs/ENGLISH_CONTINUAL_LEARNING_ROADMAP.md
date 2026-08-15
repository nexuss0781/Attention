# Attention English Continual-Learning Curriculum

## 1. Curriculum purpose

The project goal is to build an **English conversational model that understands ambiguity and responds appropriately**. The model must learn more than text continuation. It must develop English structure, semantic understanding, instruction following, conversation state, ambiguity detection, clarification, factual caution, and helpful dialogue behavior.

The curriculum is organized as **dependent chapters**. Each chapter contains modules. Each module is one competency lesson and one continual-learning session or small sequence of sessions. A later chapter cannot be promoted unless its prerequisites are accepted and its regression tests preserve the earlier chapters.

The operating principle is:

> **Train one bounded lesson, test the exact behavior taught by that lesson, stop, show the evidence, and wait for a manual promotion decision.**

This is not a final benchmark plan. It is a learning path that explains what the model is expected to learn at every stage and what evidence is required before moving forward.

## 2. Dependency structure

The dependency order is intentionally strict:

```text
Chapter 0  System readiness
    ↓
Chapter 1  Causal English foundation
    ↓
Chapter 2  Lexical, sentence, and local-coherence competence
    ↓
Chapter 3  Passage meaning and grounded English understanding
    ↓
Chapter 4  Single-turn instruction following
    ↓
Chapter 5  Multi-turn conversation state
    ↓
Chapter 6  Ambiguity detection
    ↓
Chapter 7  Clarification and conversational repair
    ↓
Chapter 8  Helpfulness, correctness, and preference alignment
    ↓
Chapter 9  Integrated English ambiguity-aware conversation
    ↓
Chapter 10 Continual regression and controlled expansion
```

A chapter is not passed because its training loss decreases. It is passed only when the module-specific behavior is demonstrated, the held-out tests are acceptable, and earlier accepted behavior remains within its regression limits.

## 3. What one session means

A **session** is the smallest auditable learning unit. It has one parent checkpoint, one competency module, one deterministic training chunk, one held-out validation chunk, one test set, and one manual decision.

| Session part | Required content |
|---|---|
| Parent | SHA-256 hash of the last accepted model and training-state checkpoint |
| Lesson | Chapter ID, module ID, competency statement, and expected behavior |
| Training chunk | Dataset ID, revision, document range, token range, tokenizer hash, and split hash |
| Training budget | Maximum tokens, steps, batch shape, learning rate, and memory budget |
| Baseline | Loss and inference behavior before the lesson |
| Training | Per-step loss, validation loss, gradient norm, learning rate, and tokens processed |
| Competency tests | Fixed prompts, expected behavior, scoring rubric, and generated outputs |
| Regression tests | Previously passed prompts and thresholds from earlier chapters |
| Artifacts | Model checkpoint, training-state checkpoint, traces, reports, outputs, and hashes |
| Decision | `PROMOTE`, `RETRY`, or `ABORT`, with the user's rationale |

The session always ends in `AWAITING_REVIEW`. The script must never promote automatically. A retry receives a new session ID and keeps the failed session immutable.

## 4. General data and resource rules

The active model must keep one fixed parameter set during a session. Continual learning controls resource usage by training on bounded chunks, streaming source data, and not accumulating one new model or adapter for every lesson. The current linear-memory attention invariant remains mandatory: no dense softmax attention matrix and no O(n²) token-pair allocation.

The English data roles are separated:

| Role | Primary source | Use |
|---|---|---|
| Broad English pretraining | FineWeb English | General syntax, vocabulary, prose, and world-text distribution |
| Explanatory quality | FineWeb-Edu | Clear explanatory and educational language after broad English is stable |
| Human instruction | English-filtered OASST1 | Human-written conversational supervision |
| Multi-turn coverage | English-filtered UltraChat | Broad instructional dialogue, used as a supplement rather than the only source |
| Preference quality | HelpSteer2 or HelpSteer2-Preference | Helpfulness, correctness, coherence, and response-quality preferences |
| Ambiguity supervision | AmbigQA plus a human-reviewed clarification set | Ambiguity recognition and interpretation separation |

Every chunk must be English-filtered, normalized, deduplicated, split without document leakage, and sealed with lineage metadata. The first token budget for a lesson is deliberately small; it increases only after the lesson's validation evidence is stable.

## 5. Chapter 0 — System readiness

### Chapter purpose

Prove that the model can be initialized, trained, evaluated, saved, reloaded, and resumed deterministically. This chapter is about the learning instrument, not English ability.

### Modules

| Module | Lesson | Prerequisite | Initial budget |
|---|---|---|---:|
| 0.1 | Numerical and tensor correctness | None | Existing unit fixtures |
| 0.2 | Tokenizer and special-token correctness | 0.1 | 4,096 tokens |
| 0.3 | Forward causal-loss correctness | 0.1, 0.2 | 4,096 tokens |
| 0.4 | Backward and optimizer learning signal | 0.3 | 4,096 tokens |
| 0.5 | Checkpoint and exact resume | 0.4 | Two interrupted runs |
| 0.6 | Session artifact and manual-review lifecycle | 0.5 | One sealed session |

### Expected evidence

The model must show finite loss, finite nonzero gradients, deterministic tokenization, decreasing loss on the fixed diagnostic data, exact model reload loss, exact training-state resume equivalence, complete lineage, and clean artifact sealing.

### Acceptance

All automated checks pass, the before/after loss is reproducible, and the model state after interruption is identical to uninterrupted training.

### Non-acceptance

A run is rejected for NaN/Inf, silent parameter reset, missing parent hash, mismatched tokenizer metadata, non-reproducible reload, or any forbidden quadratic allocation. No language-quality claim is allowed from this chapter.

## 6. Chapter 1 — Causal English foundation

### Chapter purpose

Teach the model that English text is an ordered sequence and establish a measurable next-token continuation signal.

### Module 1.1 — English byte and word-boundary continuation

**Prerequisite:** Chapter 0 complete.

**Training data:** A bounded FineWeb English chunk. Start at 64,000 training tokens and 8,000 held-out tokens; a 4,096/1,024 chunk is only a calibration run and cannot be the final chapter pass.

**Lesson:** Given an English prefix, predict the next tokens while preserving punctuation, spacing, and local word structure.

**Fixed tests:** The test prefixes are selected from held-out documents, and the actual continuation remains in the test artifact. Examples have the form:

```text
The sky is
The document explains
In the morning, the
A useful method should
```

**Expected behavior:** The model should reduce held-out next-token loss, assign probability to the actual continuation, produce valid UTF-8, and avoid one-token or one-byte collapse. The expected continuation is the held-out document continuation, not a guessed universal answer.

**Automated validation:** Training loss reduction, held-out loss reduction or bounded regression, perplexity, valid UTF-8 rate, unique-token ratio, repeated-token ratio, and exact checkpoint reload.

**Acceptance:** At least 80% of fixed prefixes avoid repeated-token collapse; the model produces multiple token types; held-out loss improves or stays within the predeclared margin; and all reload results match.

**Rejection:** Every prompt emits the same token, output is invalid or empty, training improves while held-out loss diverges, or the output cannot be reproduced.

### Module 1.2 — Short English continuation

**Prerequisite:** Module 1.1 accepted.

**Training data:** 250,000–1 million additional FineWeb English tokens with 10% held out by document.

**Lesson:** Continue one to three sentences while preserving local topic, punctuation, and basic phrase structure.

**Fixed tests:** Held-out paragraphs with prefixes of 8–32 tokens. The report includes the reference continuation, generated continuation, token loss, and human-readable output.

**Acceptance:** The output stays on the local topic, uses valid sentence boundaries, and does not regress Module 1.1.

**Rejection:** The model changes topic immediately, repeats a token loop, or improves only on one memorized document range.

### Chapter 1 promotion gate

The chapter is promoted only when Modules 1.1 and 1.2 pass on independent chunks. If the same failure appears on two clean chunks after verified tokenizer and optimizer checks, stop and investigate capacity, representation, or output-head design before Chapter 2.

## 7. Chapter 2 — Lexical, sentence, and local-coherence competence

### Chapter purpose

Move from token continuation to recognizable English sentence structure, grammar, punctuation, and short-range coherence.

### Module 2.1 — Vocabulary and phrase boundaries

**Prerequisite:** Chapter 1.

**Training data:** 1–5 million broad English FineWeb tokens, with a stable vocabulary mixture and 10% held out.

**Lesson:** Preserve common word boundaries, phrase completion, punctuation, and capitalization.

**Tests:** Prefix completion, punctuation continuation, short phrase selection, and tokenization round-trip tests.

**Acceptance:** The model preserves word and sentence boundaries more often than the Chapter 1 baseline, and invalid UTF-8 or repeated-byte behavior does not increase.

### Module 2.2 — Grammar and agreement

**Prerequisite:** Module 2.1.

**Training data:** 1–5 million English tokens, with a small human-reviewed grammatical probe set.

**Lesson:** Preserve subject–verb agreement, tense, plurality, negation, and basic clause structure.

**Tests:** Minimal pairs and sentence continuations such as:

```text
The group of students
Although the weather was difficult,
The reports from the laboratory
She opened the door and
```

**Acceptance:** The preferred continuation is grammatical on the fixed probe set, and the model does not lose Chapter 1 continuation performance.

**Rejection:** Fluent-looking but unstable grammar, incorrect agreement, or repeated memorization of the probe wording.

### Module 2.3 — Local paragraph coherence

**Prerequisite:** Modules 2.1 and 2.2.

**Training data:** 5–10 million FineWeb tokens with 10–20% FineWeb-Edu only after broad-English loss is stable.

**Lesson:** Continue a short paragraph while preserving its entities, topic, and immediate discourse direction.

**Tests:** Three- to five-sentence held-out continuation, entity consistency, topic similarity, and contradiction checks.

**Acceptance:** The generated continuation remains locally related and grammatically stable, with no regression on Chapters 0–1.

### Chapter 2 promotion gate

Promote only when the model passes phrase, grammar, and paragraph tests on at least two independent English chunks. Loss reduction without coherence is insufficient.

## 8. Chapter 3 — Passage meaning and grounded English understanding

### Chapter purpose

Teach the model to use the meaning present in a passage rather than merely producing plausible generic English.

### Module 3.1 — Entity and reference tracking

**Prerequisite:** Chapter 2.

**Training data:** 1–5 million English explanatory and narrative tokens, plus held-out passage-question examples.

**Lesson:** Track names, objects, pronouns, and simple relationships across a passage.

**Tests:** Passage followed by “who,” “what,” and reference questions; entity substitution and contradiction probes.

**Expected response:** A short answer supported by the passage.

**Rejection:** Invented entities, swapped referents, or generic answers unsupported by the text.

### Module 3.2 — Temporal order and negation

**Prerequisite:** Module 3.1.

**Training data:** 1–5 million tokens plus a reviewed temporal/negation probe set.

**Lesson:** Preserve “before,” “after,” “not,” “never,” conditional language, and causal order.

**Tests:** Short passages with questions that change meaning when negation or order is reversed.

**Acceptance:** The model preserves the passage's polarity and order, and confidence decreases when the passage does not support an answer.

### Module 3.3 — Grounded explanation

**Prerequisite:** Modules 3.1 and 3.2.

**Training data:** 5–20 million mixed FineWeb/FineWeb-Edu tokens, with 100,000–250,000 held-out grounded examples.

**Lesson:** Explain an answer using only the supplied passage when the question is passage-grounded.

**Expected response:** A concise answer with a short explanation tied to the text.

**Rejection:** Confident outside knowledge presented as if it came from the passage, or failure to acknowledge missing evidence.

### Chapter 3 promotion gate

The model must answer supported questions, preserve negation and references, and say that the passage does not establish an answer when it does not. This chapter blocks instruction tuning if the model still hallucinates from simple passages.

## 9. Chapter 4 — Single-turn instruction following

### Chapter purpose

Teach the model to interpret a complete user instruction and respond in the requested way without unnecessary clarification.

### Module 4.1 — Direct task completion

**Prerequisite:** Chapter 3.

**Supervised data:** English-filtered OASST1 plus a quality-controlled UltraChat subset. Start with 100,000 examples or 1–3 million tokens, with 10% held out by conversation tree.

**Lesson:** Complete a sufficiently specified task directly.

**Tests:**

```text
Explain photosynthesis in two sentences.
Give three causes of soil erosion as a numbered list.
Summarize this paragraph in one sentence: ...
Convert this sentence to the passive voice: ...
```

**Expected response:** A direct answer in the requested format.

**Rejection:** Ignoring the requested format, answering another question, or asking for information that is already supplied.

### Module 4.2 — Scope, format, and constraints

**Prerequisite:** Module 4.1.

**Supervised data:** 250,000–1 million additional English instruction tokens.

**Lesson:** Follow multiple constraints simultaneously, such as length, style, audience, format, and exclusions.

**Tests:** Constrained rewriting, structured output, short explanations, and “do not include” probes.

**Acceptance:** At least 80% of constraints are satisfied on the fixed test set, with factual and safety regression checks.

### Module 4.3 — Uncertainty and unsupported claims

**Prerequisite:** Module 4.2.

**Supervised data:** Human-reviewed grounded instruction examples, 100,000–500,000 tokens.

**Lesson:** Distinguish known information, provided evidence, assumptions, and missing information.

**Expected response:** Answer when supported, state an assumption when harmless, and say what is missing when necessary.

### Chapter 4 promotion gate

The model must follow direct instructions without over-clarifying, preserve passage grounding, and maintain Chapters 1–3. Preference training is not allowed to begin before this chapter passes.

## 10. Chapter 5 — Multi-turn conversation state

### Chapter purpose

Teach the model to maintain context across turns and treat the latest valid user information as active.

### Module 5.1 — Entity and referent memory

**Prerequisite:** Chapter 4.

**Supervised data:** 50,000–100,000 English multi-turn conversations, approximately 500,000–1 million tokens, with entire conversation trees held out.

**Lesson:** Resolve “it,” “that option,” “the second one,” and named entities across turns.

**Tests:** Three- to eight-turn conversations with entity introduction, reference, and correction.

**Expected response:** The model uses the correct referent without asking the user to repeat known information.

### Module 5.2 — Constraint and correction memory

**Prerequisite:** Module 5.1.

**Data:** 50,000–100,000 additional reviewed multi-turn examples.

**Lesson:** Retain user constraints, incorporate corrections, and replace obsolete information with newer valid information.

**Rejection:** Reverting to an earlier constraint, ignoring a correction, or resetting the conversation.

### Module 5.3 — Multi-turn task completion

**Prerequisite:** Modules 5.1 and 5.2.

**Data:** 100,000–500,000 tokens of multi-turn instruction tasks.

**Lesson:** Plan and complete a small task over several turns without losing the objective.

### Chapter 5 promotion gate

The model must pass state, correction, and task-completion conversations and preserve direct single-turn behavior.

## 11. Chapter 6 — Ambiguity detection

### Chapter purpose

Teach the model to recognize when a user request has multiple materially different interpretations.

### Module 6.1 — Ambiguous versus answerable classification

**Prerequisite:** Chapter 5.

**Data:** AmbigQA plus a human-reviewed English ambiguity set. Begin with approximately 14,000 AmbigQA examples and 10,000–25,000 reviewed examples; hold out entire question families.

**Lesson:** Decide whether the request is sufficiently specified to answer or requires clarification.

**Tests:** Paired prompts where one version is ambiguous and the other adds the missing constraint.

**Examples:**

```text
What is the best Python library for this?
What is the best Python library for sparse CPU linear algebra?
Book a flight to Portland next Friday.
Book a flight to Portland, Oregon, next Friday after 6 PM.
```

**Expected response:** The first and third prompts are recognized as underspecified; the second and fourth are treated as sufficiently specified.

**Rejection:** Always asking, never asking, or treating a harmless wording variation as a blocking ambiguity.

### Module 6.2 — Missing-variable identification

**Prerequisite:** Module 6.1.

**Data:** 25,000–100,000 reviewed ambiguity examples.

**Lesson:** Identify the smallest missing decision that changes the answer, such as location, time, account, unit, audience, or desired output format.

**Expected response:** One concise clarification focused on the missing variable.

### Chapter 6 promotion gate

The model must classify ambiguous and answerable prompts correctly and identify the relevant missing variable without inventing one.

## 12. Chapter 7 — Clarification and conversational repair

### Chapter purpose

Teach the model to ask useful clarification questions, use the user's answer, and continue the task.

### Module 7.1 — Minimal clarification

**Prerequisite:** Chapter 6.

**Data:** 25,000–100,000 clarification conversations derived from AmbigQA and human-reviewed English examples.

**Lesson:** Ask one concise question when one question resolves the ambiguity.

**Example:**

```text
User: Book a flight to Portland next Friday.
Expected clarification: Do you mean Portland, Oregon, or Portland, Maine?
```

**Rejection:** Asking five unrelated questions, guessing the location, or refusing to help.

### Module 7.2 — Clarification answer integration

**Prerequisite:** Module 7.1.

**Data:** 50,000–150,000 multi-turn clarification-and-resolution examples.

**Lesson:** Use the user's clarification and complete the original request without asking the same question again.

**Example:**

```text
User: Portland, Oregon, after 6 PM.
Expected behavior: Continue using Portland, Oregon, and the after-6-PM constraint.
```

### Module 7.3 — Assumption versus clarification

**Prerequisite:** Module 7.2.

**Data:** 25,000–75,000 reviewed examples.

**Lesson:** State a harmless assumption when clarification would add friction, but ask when interpretations materially change the result.

**Chapter acceptance:** The model asks when needed, answers directly when possible, states assumptions clearly, and uses the answer to continue.

## 13. Chapter 8 — Helpfulness, correctness, and preference alignment

### Chapter purpose

Improve response quality after the model has learned when to answer and when to clarify.

### Module 8.1 — Correctness and evidence preference

**Prerequisite:** Chapter 7.

**Preference data:** HelpSteer2 or HelpSteer2-Preference. Start with 25,000–100,000 English preference comparisons, approximately 250,000–1 million preference tokens, with a held-out set.

**Lesson:** Prefer the answer that is more correct, supported, coherent, and appropriately cautious.

### Module 8.2 — Helpfulness without overconfidence

**Prerequisite:** Module 8.1.

**Data:** 25,000–100,000 preference comparisons involving uncertainty, refusal, clarification, and concise explanations.

**Lesson:** Be useful without inventing facts, being sycophantic, or refusing unnecessarily.

### Module 8.3 — Regression-preserving preference update

**Prerequisite:** Module 8.2.

**Data:** A replay mixture containing 10–20% earlier instruction, ambiguity, and clarification examples.

**Acceptance:** Preference optimization improves response quality without reducing ambiguity detection, clarification precision, direct answering, or groundedness.

## 14. Chapter 9 — Integrated ambiguity-aware English conversation

### Chapter purpose

Combine the complete capability into the final target behavior.

### Module 9.1 — Interaction-mode selection

**Prerequisite:** Chapters 1–8.

**Lesson:** Choose among four actions: answer directly, ask one clarification, state an assumption, or decline unsupported action.

**Test families:**

| Situation | Correct mode |
|---|---|
| Fully specified factual question | Answer directly |
| Materially ambiguous request | Ask one clarification |
| Harmless underspecification | State an assumption and proceed |
| Unsupported or unavailable information | State the limitation rather than inventing |

### Module 9.2 — Multi-turn ambiguity resolution

**Prerequisite:** Module 9.1.

**Lesson:** Detect ambiguity, ask the smallest useful question, use the answer, and complete the task while preserving all previous constraints.

### Module 9.3 — Final conversational regression

**Prerequisite:** Module 9.2.

**Data:** 1–5 million tokens in a replay mixture: approximately 50% instruction/conversation, 20% ambiguity and clarification, 20% grounded semantic examples, and 10% broad English replay.

**Acceptance:** The model demonstrates the expected interaction mode on fixed test families, preserves earlier chapters, and produces human-acceptable responses across multiple independent sessions.

**Final rejection:** Any systematic behavior of always answering, always clarifying, confidently inventing missing details, losing conversation state, or regressing on direct instruction blocks the final claim.

## 15. Chapter 10 — Continual maintenance and expansion

### Chapter purpose

Keep the accepted model stable while introducing new English data, domains, and capability lessons.

### Modules

| Module | Purpose |
|---|---|
| 10.1 | Replay earlier competency examples during every later session |
| 10.2 | Add new English domains through bounded independent chunks |
| 10.3 | Detect regression after each new data source |
| 10.4 | Compare architecture and resource changes before expansion |
| 10.5 | Periodically consolidate and archive accepted checkpoints |

No new dataset is allowed to silently change the tokenizer, output vocabulary, loss definition, or accepted competency rubric. Any such change begins a new curriculum branch with an explicit migration record.

## 16. Acceptance and failure protocol

Each module produces a report with these sections:

1. **Baseline:** parent checkpoint loss and pre-training inference outputs.
2. **Lesson:** the exact behavior this session was supposed to teach.
3. **Training evidence:** token budget, steps, learning-rate schedule, loss curve, validation curve, and gradient diagnostics.
4. **Competency evidence:** fixed prompts, expected behavior, generated outputs, and scores.
5. **Regression evidence:** earlier module tests and changes from baseline.
6. **Resource evidence:** parameter count, active memory, training time, and confirmation of linear-memory attention.
7. **Verdict recommendation:** automated recommendation only; no automatic promotion.

The user then chooses:

| Verdict | Meaning |
|---|---|
| `PROMOTE` | The module's competency and regression gates are met; use this checkpoint as the next parent |
| `RETRY` | The objective is valid but the chunk, budget, or optimization was insufficient; create a new session ID |
| `ABORT` | Evidence indicates a data, tokenizer, objective, optimization, or architecture problem; stop and investigate |

A single failed session is not an architecture verdict. The same module must fail on at least two independent clean chunks after data, tokenization, target construction, and gradient checks pass. Only then is an architecture review justified.

## 17. First session to approve

The first meaningful session after system readiness is **Chapter 1, Module 1.1: English byte and word-boundary continuation**. Before implementation begins, its session manifest must specify:

| Field | Required decision |
|---|---|
| Parent | Accepted compatible model checkpoint |
| Data | FineWeb English configuration, revision, document range, and English filter |
| Train budget | Start at 64,000 tokens; use 4,096 only for calibration |
| Validation budget | 8,000 tokens, or 1,024 only for calibration |
| Tests | Held-out prefixes with actual continuations, not unrelated generic prompts |
| Pass | Loss, valid UTF-8, non-degenerate token diversity, continuation quality, reload equality, and no regression |
| Stop | Seal artifacts and wait for the user's manual verdict |

The model is not allowed to advance to Chapter 2 because its loss decreased once. It advances only after the user reviews the Chapter 1 evidence and accepts the demonstrated competency.

## References

[1]: https://aclanthology.org/2020.emnlp-main.466/ "AmbigQA: Answering Ambiguous Open-domain Questions"
[2]: https://huggingface.co/datasets/nvidia/HelpSteer2 "HelpSteer2 dataset card"
[3]: https://huggingface.co/datasets/HuggingFaceFW/fineweb "FineWeb dataset card"
[4]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu "FineWeb-Edu dataset card"
[5]: https://aclanthology.org/2024.acl-long.840/ "Dolma: An Open Corpus of Three Trillion Tokens"
[6]: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 "DCLM-Baseline dataset card"
[7]: https://huggingface.co/datasets/OpenAssistant/oasst1 "OpenAssistant OASST1 dataset card"
[8]: https://aclanthology.org/2023.emnlp-main.183/ "UltraChat paper"
