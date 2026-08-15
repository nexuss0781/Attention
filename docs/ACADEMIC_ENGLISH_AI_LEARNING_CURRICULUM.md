# Academic English AI Learning Curriculum

## Overview

This curriculum defines a progressive learning path for an English conversational AI system. The objective is to move from basic English sequence learning to reliable conversational ambiguity understanding.

The curriculum is competency-based. Each module teaches one class of abilities. Each module contains submodules that move from basic forms to more complex use. A module is complete only when the model demonstrates the intended competency in new examples, not merely examples it has seen before.

The learning path follows this order:

1. Learn English sequences.
2. Learn words, phrases, and sentence structure.
3. Learn paragraph meaning and relationships between ideas.
4. Learn factual and evidence-based understanding.
5. Learn to follow explicit instructions.
6. Learn to maintain conversational context.
7. Learn to recognize ambiguity.
8. Learn to ask useful clarification questions.
9. Learn to reason under incomplete information.
10. Learn helpful, accurate, and appropriately cautious conversation.
11. Integrate all abilities into natural English dialogue.

A later module depends on the successful completion of earlier modules. The model should not be advanced simply because its general language score improves. It must demonstrate the specific ability taught by the current module.

## Learning progression model

Every module follows the same academic pattern:

- **Competency:** The ability the model must acquire.
- **Goal:** The learning outcome for the module.
- **Submodules:** The ordered concepts and behaviors taught inside the module.
- **Practice:** The type of examples used to develop the ability.
- **Assessment:** The examples used to determine whether the ability transferred to new cases.
- **Mastery:** The minimum observable behavior required before progression.
- **Failure response:** The next learning action when mastery is not achieved.

The model progresses only when it can:

- Demonstrate the current competency on unseen examples.
- Preserve the competencies learned earlier.
- Produce stable behavior across different examples of the same type.
- Avoid replacing understanding with repetition, guessing, or superficial pattern matching.

# Module 1 — English Sequence Learning

## Competency

The model understands English as an ordered sequence of symbols, words, phrases, and sentences.

## Goal

The model learns to predict plausible subsequent English text while preserving local spelling, spacing, punctuation, and phrase structure.

## Submodules

### 1.1 Symbol and byte regularity

- Learn common English character and byte sequences.
- Distinguish letters, spaces, punctuation, digits, and line boundaries.
- Preserve valid text encoding.
- Avoid producing a single repeated symbol as a general response.

### 1.2 Word-boundary continuation

- Learn common prefixes and suffixes.
- Predict likely word completions.
- Preserve spaces between words.
- Distinguish punctuation attached to a word from punctuation beginning a new phrase.

### 1.3 Phrase continuation

- Complete common noun phrases, verb phrases, and prepositional phrases.
- Preserve short-range grammatical expectations.
- Continue a phrase without changing its subject unnecessarily.

### 1.4 Sentence continuation

- Continue a sentence after a short English prefix.
- Preserve punctuation and sentence boundaries.
- Produce more than a repeated-token loop.

## Competency examples

- `The sky is ...`
- `A small child opened ...`
- `The purpose of this method is ...`
- `In the morning, the researchers ...`

## Mastery requirement

The model must generate valid, varied English continuations and demonstrate improved next-token prediction on unseen English examples. It must not merely reduce training loss while producing repeated or invalid output.

## Dependency

This module is the prerequisite for all later modules.

# Module 2 — Vocabulary and Phrase Formation

## Competency

The model represents and combines English words and phrases in context.

## Goal

The model learns that the meaning and grammatical role of a word depend partly on the words surrounding it.

## Submodules

### 2.1 Vocabulary expansion

- Learn common nouns, verbs, adjectives, adverbs, and function words.
- Distinguish frequent words from rare but meaningful words.
- Preserve capitalization and common morphological forms.

### 2.2 Morphology

- Learn singular and plural forms.
- Learn tense and aspect changes.
- Learn common prefixes, suffixes, and derivations.

### 2.3 Phrase composition

- Combine modifiers with nouns.
- Combine subjects with verbs and objects.
- Distinguish similar phrases with different meanings.

### 2.4 Contextual word choice

- Select a word that fits the local sentence context.
- Distinguish words with multiple meanings when nearby context provides evidence.
- Avoid replacing a contextually required word with a frequent but incorrect word.

## Competency examples

- `The scientist conducted a ...`
- `The children were ...`
- `The report describes several ...`
- `She placed the book on the ...`

## Mastery requirement

The model selects contextually suitable words and preserves phrase meaning across unseen examples.

## Dependency

Requires Module 1.

# Module 3 — Grammar and Sentence Structure

## Competency

The model understands basic English grammatical structure.

## Goal

The model produces and interprets sentences with stable agreement, tense, negation, clause structure, and punctuation.

## Submodules

### 3.1 Subject–verb agreement

- Match singular subjects with singular verbs.
- Match plural subjects with plural verbs.
- Avoid being misled by intervening nouns.

### 3.2 Tense and aspect

- Distinguish past, present, and future events.
- Preserve ongoing and completed actions.
- Maintain tense across a sentence and short paragraph.

### 3.3 Negation

- Preserve `not`, `never`, `no`, and related negative constructions.
- Distinguish affirmative and negative meanings.
- Avoid reversing the meaning of a question or statement.

### 3.4 Clause and sentence boundaries

- Recognize independent and dependent clauses.
- Use conjunctions appropriately.
- Preserve commas, periods, question marks, and quotation marks.

### 3.5 Grammatical transformation

- Form questions from statements.
- Change active voice to passive voice.
- Change singular forms to plural forms.
- Preserve meaning during transformation.

## Competency examples

- `The group of students ...`
- `Although the weather was difficult, ...`
- `The reports from the laboratory ...`
- `She opened the door and ...`

## Mastery requirement

The model consistently preserves grammatical relations and meaning in new sentences and minimal-pair tests.

## Dependency

Requires Module 2.

# Module 4 — Local Coherence and Paragraph Structure

## Competency

The model maintains a topic, entities, and logical continuity across several sentences.

## Goal

The model can continue or summarize a short paragraph without immediately changing topic, contradicting earlier statements, or losing the main subject.

## Submodules

### 4.1 Topic continuity

- Identify the central topic of a paragraph.
- Continue writing about that topic.
- Avoid unrelated or abrupt topic shifts.

### 4.2 Entity continuity

- Preserve names, objects, places, and roles.
- Avoid changing an entity's attributes without evidence.
- Track references such as `it`, `they`, `this method`, and `the earlier result`.

### 4.3 Event continuity

- Preserve the order of events.
- Avoid introducing events that contradict the paragraph.
- Connect causes and consequences appropriately.

### 4.4 Paragraph completion

- Continue a paragraph for several sentences.
- Preserve tone and grammatical style.
- End with a plausible local conclusion when appropriate.

## Competency examples

- Continue a three-sentence scientific paragraph.
- Continue a short narrative after a named character acts.
- Identify which earlier noun a pronoun refers to.
- Summarize the paragraph in one sentence.

## Mastery requirement

The model maintains local topic and entity consistency across unseen paragraphs and does not rely on copying a memorized passage.

## Dependency

Requires Module 3.

# Module 5 — Semantic Comprehension

## Competency

The model extracts and relates meaning from an English passage.

## Goal

The model answers questions using information expressed in the passage and distinguishes supported information from unsupported guesses.

## Submodules

### 5.1 Entity and reference understanding

- Identify who or what performed an action.
- Resolve pronouns and noun references.
- Track relationships between entities.

### 5.2 Attribute understanding

- Identify properties, quantities, locations, and conditions.
- Distinguish an entity from a similar entity.
- Preserve whether an attribute applies or does not apply.

### 5.3 Temporal understanding

- Identify what happened first, next, and last.
- Distinguish past, present, and planned events.
- Preserve temporal words such as `before`, `after`, and `during`.

### 5.4 Negation and contrast

- Preserve negative statements.
- Distinguish alternatives and contrasts.
- Avoid answering the opposite of what the passage states.

### 5.5 Passage-based answering

- Answer only from information supported by the passage.
- State when the passage does not provide enough information.
- Give a concise explanation for the answer.

## Competency examples

- `Who performed the experiment?`
- `What happened before the meeting?`
- `Which object was not included?`
- `Does the passage establish the cause?`

## Mastery requirement

The model answers supported questions accurately, preserves negation and reference, and does not present outside guesses as passage evidence.

## Dependency

Requires Module 4.

# Module 6 — Factual and Explanatory Understanding

## Competency

The model explains concepts and relationships in clear English while distinguishing evidence from uncertainty.

## Goal

The model gives a short, coherent explanation that is relevant to the question and appropriately qualified when evidence is incomplete.

## Submodules

### 6.1 Definition

- State what a concept means.
- Distinguish a definition from an example.
- Avoid circular or empty explanations.

### 6.2 Cause and effect

- Identify stated causes and consequences.
- Avoid treating correlation as proven causation.
- Explain a relationship in the correct direction.

### 6.3 Comparison

- Identify similarities and differences.
- Compare the same attributes across objects or concepts.
- Avoid changing the comparison criteria midway.

### 6.4 Explanation length and structure

- Give a short answer when requested.
- Expand when explanation is requested.
- Organize multiple points clearly.

### 6.5 Uncertainty

- Separate known information from assumptions.
- State when evidence is insufficient.
- Avoid confident invention.

## Competency examples

- `Explain photosynthesis in two sentences.`
- `Why does the method reduce memory use?`
- `How are these two approaches different?`
- `What can be concluded from this passage?`

## Mastery requirement

The model gives relevant, coherent, evidence-aware explanations and does not replace missing information with confident fabrication.

## Dependency

Requires Module 5.

# Module 7 — Instruction Following

## Competency

The model follows a user's explicit request and satisfies its constraints.

## Goal

The model answers directly when the instruction is sufficiently specified and produces the requested form of output.

## Submodules

### 7.1 Direct task completion

- Answer a clear question.
- Perform a requested transformation.
- Summarize supplied text.
- Extract requested information.

### 7.2 Output format

- Produce lists, tables, steps, paragraphs, or structured fields as requested.
- Preserve the requested order.
- Avoid adding unrelated material.

### 7.3 Constraint following

- Respect length limits.
- Respect audience and tone.
- Include required elements.
- Exclude prohibited elements.

### 7.4 Instruction priority

- Follow the latest valid user instruction.
- Distinguish the main task from examples and quoted text.
- Avoid treating explanatory text as a new task.

## Competency examples

- `Explain photosynthesis in two sentences.`
- `Give three causes as a numbered list.`
- `Summarize this passage in one sentence.`
- `Rewrite this paragraph for a child.`

## Mastery requirement

The model completes clear instructions directly, satisfies the requested format, and does not ask unnecessary questions.

## Dependency

Requires Module 6.

# Module 8 — Conversational State

## Competency

The model maintains information across multiple conversational turns.

## Goal

The model remembers relevant entities, constraints, corrections, and prior decisions while responding to the latest request.

## Submodules

### 8.1 Entity memory

- Remember names, objects, places, and roles introduced earlier.
- Resolve references such as `it`, `that one`, and `the second option`.

### 8.2 Constraint memory

- Preserve user preferences and requirements.
- Apply earlier constraints to later actions.
- Avoid dropping important conditions.

### 8.3 Correction handling

- Accept a user's correction.
- Replace outdated information with the corrected information.
- Avoid returning to the earlier error.

### 8.4 Conversational task continuity

- Maintain the original objective across turns.
- Ask only for information that is still missing.
- Complete the task after the required information is provided.

## Competency examples

- A user introduces a project name and later says `improve it`.
- A user chooses the second option and later asks for its disadvantages.
- A user corrects a date, location, or quantity.
- A user changes the required output format midway through the conversation.

## Mastery requirement

The model preserves the relevant state, incorporates corrections, and does not reset the conversation after each turn.

## Dependency

Requires Module 7.

# Module 9 — Ambiguity Recognition

## Competency

The model distinguishes a sufficiently specified request from a request with multiple materially different interpretations.

## Goal

The model recognizes when answering immediately would require an unjustified assumption.

## Submodules

### 9.1 Ambiguity detection

- Identify when a request has more than one plausible interpretation.
- Distinguish real ambiguity from harmless variation.
- Recognize when the answer changes depending on the interpretation.

### 9.2 Missing-information identification

- Identify the missing location, time, object, account, unit, audience, or desired output.
- Select the missing variable that matters most.

### 9.3 Ambiguous versus answerable distinction

- Answer a request that contains enough information.
- Avoid asking for information already provided.
- Avoid assuming information that materially changes the answer.

## Competency examples

- `What is the best Python library for this?`
- `What is the best Python library for sparse CPU linear algebra?`
- `Book a flight to Portland next Friday.`
- `Book a flight to Portland, Oregon, next Friday after 6 PM.`
- `How do I reset it?`
- `How do I reset the administrator password on Ubuntu 24.04?`

## Expected behavior

- The first, third, and fifth requests require clarification.
- The second, fourth, and sixth requests are sufficiently specified for a direct answer or next action.

## Mastery requirement

The model asks for clarification only when materially different interpretations exist and identifies the correct missing variable.

## Dependency

Requires Module 8.

# Module 10 — Clarification and Conversational Repair

## Competency

The model asks a useful clarification question, receives the answer, and continues the task correctly.

## Goal

The model reduces uncertainty with the smallest useful question rather than guessing, over-questioning, or refusing.

## Submodules

### 10.1 Minimal clarification

- Ask one question when one missing detail is decisive.
- Use concrete alternatives when appropriate.
- Avoid asking several unrelated questions at once.

### 10.2 Clarification quality

- Make the question easy for the user to answer.
- Explain why the distinction matters when necessary.
- Avoid vague questions such as `Can you clarify?` when a precise question is possible.

### 10.3 Answer integration

- Use the user's answer.
- Continue the original task.
- Do not ask the same question again.

### 10.4 Assumption versus clarification

- State a harmless assumption when it does not materially affect the result.
- Ask when different interpretations produce different outcomes.

## Competency example

```text
User: Book a flight to Portland next Friday.
Model: Do you mean Portland, Oregon, or Portland, Maine?
User: Portland, Oregon, after 6 PM.
Model: Continue using Portland, Oregon, and the after-6-PM constraint.
```

## Mastery requirement

The model asks one useful clarification, uses the answer correctly, and completes the task without inventing missing details.

## Dependency

Requires Module 9.

# Module 11 — Reasoning Under Incomplete Information

## Competency

The model decides whether to answer, ask, assume, or state uncertainty when information is incomplete.

## Goal

The model chooses an appropriate response mode instead of applying one behavior to every situation.

## Submodules

### 11.1 Evidence and conclusion

- Distinguish evidence from interpretation.
- Separate what is known from what is inferred.
- State the strength of the conclusion.

### 11.2 Safe assumptions

- Identify assumptions that do not materially change the result.
- State assumptions transparently.
- Avoid hidden assumptions that change the task.

### 11.3 Information sufficiency

- Determine whether the available information is enough.
- Identify the smallest missing fact that would resolve uncertainty.

### 11.4 Response-mode selection

Choose among:

- Answer directly.
- Ask one clarification.
- State an assumption and answer.
- State that the information is insufficient.

## Mastery requirement

The model chooses the correct response mode and explains uncertainty without becoming evasive or overconfident.

## Dependency

Requires Module 10.

# Module 12 — Helpfulness and Correctness

## Competency

The model gives useful, accurate, relevant, and appropriately cautious responses.

## Goal

The model balances completeness, clarity, correctness, brevity, uncertainty, and user intent.

## Submodules

### 12.1 Factual correctness

- Prefer supported answers.
- Avoid invented facts.
- Correct known errors when they are identified.

### 12.2 Relevance

- Address the user's actual goal.
- Remove irrelevant explanation.
- Preserve required context.

### 12.3 Appropriate detail

- Be concise for simple questions.
- Explain when the question requires reasoning.
- Avoid unnecessary verbosity.

### 12.4 Uncertainty and limitations

- State uncertainty when evidence is weak.
- Avoid pretending to have performed an unavailable action.
- Distinguish inability from refusal.

### 12.5 Helpful correction

- Correct a mistaken premise respectfully.
- Explain the important distinction.
- Continue helping after the correction.

## Mastery requirement

The model is helpful without becoming overconfident, verbose, evasive, or sycophantic.

## Dependency

Requires Modules 6, 7, 10, and 11.

# Module 13 — Integrated Ambiguity-Aware Conversation

## Competency

The model combines English understanding, instruction following, conversation state, ambiguity recognition, clarification, reasoning, and helpfulness.

## Goal

The model behaves like a reliable English conversational assistant across complete multi-turn tasks.

## Submodules

### 13.1 Interaction-mode selection

Select the correct action:

- Answer directly when the request is clear.
- Ask one clarification when ambiguity changes the answer.
- State an assumption when clarification is unnecessary.
- State a limitation when the answer is unsupported.

### 13.2 Multi-turn ambiguity resolution

- Detect ambiguity.
- Ask a minimal clarification.
- Use the user's answer.
- Complete the task.
- Preserve earlier constraints.

### 13.3 Cross-domain transfer

Apply the same behavior to:

- General knowledge.
- Technical questions.
- Planning.
- Writing and editing.
- Data interpretation.
- Everyday requests.

### 13.4 Conversational quality

- Remain clear and respectful.
- Avoid unnecessary repetition.
- Preserve the user's intent.
- Use a natural amount of detail.

## Final competency examples

### Clear request

```text
User: Explain photosynthesis in two sentences.
Expected behavior: Answer directly in approximately two sentences.
```

### Materially ambiguous request

```text
User: What is the best library for this?
Expected behavior: Ask what task, language, platform, or constraint determines the choice.
```

### Clarified request

```text
User: I need a Python library for sparse CPU linear algebra.
Expected behavior: Answer directly with a relevant recommendation and explanation.
```

### Incomplete evidence

```text
User: Which approach is faster based on this short description?
Expected behavior: State what can and cannot be concluded from the supplied information.
```

## Final mastery requirement

The model must select the correct interaction mode, produce useful English, preserve conversation state, clarify material ambiguity, avoid unnecessary clarification, and maintain earlier competencies across unseen multi-turn conversations.

## Dependency

Requires Modules 1–12.

# Final progression rule

The model advances one module at a time:

- A module begins with a clearly defined competency.
- The model practices that competency on selected learning examples.
- The model is evaluated on unseen examples of the same competency.
- Earlier competencies are checked again.
- The user reviews the evidence.
- The user chooses whether the model advances, repeats the lesson, or stops for diagnosis.

The curriculum is complete only when Module 13 is mastered. Lower loss alone never constitutes completion.
