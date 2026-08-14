# Attention Sequential Training Dataset Curriculum

The project should not train on every source simultaneously. The recommended procedure is **sequential checkpointed training**, where each stage has one purpose, one frozen dataset manifest, one tokenizer identity, and one reloadable checkpoint. The first objective is to establish a measurable learning signal on a manageable subset; scaling and long-context evaluation come later.

## Selected curriculum

| Stage | Selected dataset | Use | Why selected | Boundary |
|---|---|---|---|---|
| 0 | Small reviewed subset of FineWeb2 | Pipeline and loss debugging, then short continued pretraining | Multilingual, versioned, ODC-BY, and exposes manageable language subsets and reproducible file paths [1] | Save `stage0_debug` checkpoint; do not report quality |
| 1 | FineWeb2 English subset | General-language continued pretraining | Better practical fit than downloading multi-trillion-token corpora; lets the project control token budget and shard size [1] | Save `stage1_general` checkpoint and record exact subset/checksums |
| 2 | Addis AI Wikipedia Amharic | Amharic continued pretraining/adaptation | Apache-2.0 dataset with Amharic-English parallel Wikipedia text and metadata; use the Amharic side for language adaptation rather than treating it as chat data [2] | Save `stage2_amharic` checkpoint; evaluate English retention and Amharic perplexity separately |
| 3 | OpenAssistant OASST1 | Supervised conversational fine-tuning | Apache-2.0, human conversation records, multilingual fields, review results, toxicity scores, and filtering labels [3] | Save `stage3_oasst` checkpoint; filter and format only selected conversation branches |
| 4 | Aya Dataset language-filtered subset, only after exact artifact/license validation | Multilingual instruction fine-tuning, especially low-resource-language coverage | Human-curated instruction-following data spanning 65 languages; the associated collection covers more languages and tasks [4] [5] | Save `stage4_aya` checkpoint only after manifest license and language checks pass |
| 5 | Project-specific permitted instruction set | Narrow task or teacher-interface fine-tuning | Aligns the model with the actual intended interface while keeping the source and prompt policy inspectable | Save final task checkpoint; keep evaluation data separate |

## What not to use for training

PG-19 should be reserved initially for held-out long-context evaluation and perplexity/NLL comparison, not mixed into the first training stages. Its official description identifies it as a benchmark built from historical Project Gutenberg books and warns that its dated language and historical biases make it unsuitable as the sole source for a general-purpose production dialogue model [6]. Aya Evaluation Suite should remain evaluation-only. The OASST1 and Aya red-teaming resources should be used for safety evaluation or filtering tests, not blindly merged into ordinary supervised fine-tuning.

Dolma remains a strong large-scale pretraining reference, but its full corpus is operationally excessive for the current environment. The official card reports a 16.4 GB roughly 10-billion-token sample for exploration and a 4.5 TB v1.7 release [7]. It should be a later scale-up option, not the first local training dependency.

## Immediate training procedure

Begin with Stage 0 using a small fixed subset and a short context length. Verify that the training loss decreases over multiple checkpoints, that validation loss is finite, and that reloading a checkpoint reproduces the next loss exactly. Then continue from that checkpoint into Stage 1 without resetting the optimizer or tokenizer metadata unless the run manifest explicitly records the change. Stage 2 should use a new optimizer schedule only if the manifest records the reason; the model weights and tokenizer remain continuous. Stages 3 and 4 should use supervised causal formatting and separate validation sets.

Every stage must store the dataset identifier, subset or language configuration, dataset revision, source checksums, tokenizer version, architecture serialization, code commit, seed, optimizer state, and checkpoint parent. A stage cannot start if its tokenizer metadata does not match the parent checkpoint or if its source manifest contains unresolved license status.

The first real quality gate is not 100M-token context. It is a controlled small run where training loss falls, held-out loss is reproducible after reload, and the sequential transition from Stage 0 to Stage 1 changes the expected loss in the correct direction. Only after that gate should the project invest in analytical backward kernels and larger corpora.

## References

[1]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 "FineWeb2 official dataset card"

[2]: https://huggingface.co/datasets/addisai/wikipedia-amharic "Addis AI Amharic Wikipedia dataset card"

[3]: https://huggingface.co/datasets/OpenAssistant/oasst1 "OpenAssistant OASST1 official dataset card"

[4]: https://aclanthology.org/2024.acl-long.620/ "Aya Dataset paper"

[5]: https://huggingface.co/collections/CohereLabs/aya-datasets "Cohere Labs Aya datasets collection"

[6]: https://github.com/google-deepmind/pg19 "PG-19 official repository"

[7]: https://huggingface.co/datasets/allenai/dolma "Dolma official dataset card"
