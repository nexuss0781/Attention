# Dataset Selection Research Findings

Dolma is described by the official dataset card as a 3-trillion-token corpus mixing web content, academic publications, code, books, and encyclopedic material. The card states that Dolma v1.7 is available under ODC-BY, with a 16.4 GB v1.6 sample of roughly 10 billion tokens suitable for exploration. It is primarily English and has substantial curation tooling, but the full corpus is far beyond local staged-training needs. Source: https://huggingface.co/datasets/allenai/dolma

FineWeb2 is listed under ODC-BY and provides multilingual subsets with detailed versioning, file paths, and language-specific subsets. The official card exposes manageable subsets ranging from hundreds of rows to very large language collections, which makes it a practical candidate for selecting a small language-balanced slice rather than downloading the full corpus. Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

These are candidate findings only; licensing and source terms still need to be recorded in the project manifest before use, and neither dataset should be treated as automatically approved for training merely because its dataset card is public.

OpenAssistant OASST1 is marked Apache-2.0 on its official dataset card, includes English, Spanish, Russian, and more languages, and provides conversation records with review counts, review results, toxicity scores, and filtering labels. It is a strong candidate for a supervised conversation fine-tuning stage after safety filtering and language/domain selection. Source: https://huggingface.co/datasets/OpenAssistant/oasst1

The Addis AI Amharic Wikipedia dataset is marked Apache-2.0 on its official card and contains an English-Amharic parallel corpus with metadata and Wikipedia structure. It is appropriate for Amharic continued pretraining or translation adaptation, not as a conversational instruction set. Source: https://huggingface.co/datasets/addisai/wikipedia-amharic

The selection should therefore separate continued pretraining from supervised fine-tuning: use a permissively licensed multilingual/prose corpus for language adaptation, then OASST1-like reviewed conversations for instruction tuning, with the Amharic Wikipedia corpus as a distinct language-adaptation stage rather than mixing it into chat fine-tuning.

The official Cohere Labs Aya page identifies Aya as a multilingual research effort, but the page itself is high-level; the dataset card and paper should be the authoritative records for exact language and license details. The Aya dataset paper describes a human-curated multilingual instruction resource, making it a strong candidate for a multilingual fine-tuning stage if the exact released subset and license metadata are frozen in the manifest. Sources: https://cohere.com/research/aya and https://aclanthology.org/2024.acl-long.620/

The recommended fine-tuning order is therefore: first a small reviewed OpenAssistant subset for general conversation and safety labels; then an Amharic-capable Aya subset only if its exact released artifact, language coverage, and license are verified; and finally a project-specific held-out instruction set generated from permitted, inspectable sources. The project should not mix evaluation corpora into any of these fine-tuning stages.
