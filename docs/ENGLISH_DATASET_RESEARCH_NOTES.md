# English Dataset Research Notes

## Sources reviewed

1. FineWeb primary report: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
2. Dolma ACL paper: https://aclanthology.org/2024.acl-long.840/

## Findings

Dolma is described by its ACL abstract as a three-trillion-token English corpus assembled from diverse web content, scientific papers, code, public-domain books, social media, and encyclopedic materials. The paper emphasizes documented curation practices and an open data-curation toolkit.

FineWeb is the relevant web-pretraining family to compare against. The FineWeb source must be treated as a corpus family rather than assumed English-only; the planned run should select an explicit English subset or apply a deterministic language-identification filter and record the resulting kept/rejected counts and checksums. FineWeb-Edu is a candidate quality-focused subset, but it should be treated as pretraining text, not as a substitute for instruction or preference data.

The project needs two separate data roles: broad English pretraining for language modeling and English conversational/instruction data for clarification behavior. A web corpus alone is insufficient to create ChatGPT-like ambiguity clarification behavior.

## Additional sources reviewed

3. DCLM-Baseline dataset card: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
4. OpenAssistant OASST1 dataset card: https://huggingface.co/datasets/OpenAssistant/oasst1

## Additional findings

DCLM-Baseline is presented as a large Common Crawl-derived pretraining corpus with heuristic cleaning, Bloom-filter deduplication, and model-based filtering using an instruction-formatted-data classifier. The dataset card exposes language-score metadata in its examples and identifies the dataset as English, making it a candidate for English-focused pretraining. The project should still enforce its own exact language threshold, quality controls, checksums, and licensing/data-use review rather than treating the card label as sufficient.

OASST1 is tagged as English plus many other languages, Apache-2.0, human-feedback, and includes conversation data. It is therefore more relevant to instruction and preference-style training than to broad next-token pretraining. An English-only branch should filter on the message language field and retain conversation trees/ranking information where available, while avoiding mixing non-English examples into an English-focused run.

FineWeb-Edu is explicitly tagged English and described as educational web data filtered from FineWeb. Its dataset card states ODC-By v1.0 licensing and that Common Crawl terms also apply. It is a strong candidate for the first English pretraining stage when the project can accept Common Crawl lineage and can preserve the exact subset/version and document metadata.

UltraChat's EMNLP paper describes 1.5 million high-quality multi-turn instructional dialogues generated through an iterative framework, with broad topic and instruction coverage. It is useful for conversational supervised fine-tuning, but the paper explicitly says it does not involve human queries; therefore it should not be the only source for natural user ambiguity, uncertainty, or clarification behavior.

HelpSteer2 is tagged English, human-feedback, and CC-BY-4.0. Its dataset card presents it as an open helpfulness dataset for aligning models toward helpful, factually correct, coherent, and controllably complex/verbose responses. It is therefore a post-training preference/alignment candidate rather than a pretraining corpus.

AmbigQA is an English ambiguity-focused QA task with 14,042 questions. Its supervision includes finding plausible answers and rewriting questions to resolve ambiguity. It is particularly relevant for teaching and evaluating ambiguity recognition, but it is small and task-specific; it should be used as a targeted clarification curriculum/evaluation component rather than the main conversational corpus.

FineWeb2's official card is multilingual and organized by ISO language-script subsets. The card exposes many language subsets, while the direct FineWeb card is explicitly tagged English and ODC-By. For an English-only project, the cleaner choice is to use the dedicated FineWeb/FineWeb-Edu English releases rather than downloading FineWeb2 and attempting to reconstruct English from a multilingual mixture. FineWeb2 can remain useful later only if multilingual expansion becomes an explicit project goal.
