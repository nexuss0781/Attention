# Attention Stage 4 Data Policy

This policy governs any corpus used for training, validation, testing, or long-context quality claims. It is an engineering and provenance contract, not a substitute for jurisdiction-specific legal review.

## Permitted sources

A source is permitted only when its identifier, owner or provider, license or permission basis, retrieval timestamp, checksum, intended split, and permitted use are recorded in the versioned dataset manifest. Public availability alone is not sufficient. PG-19 may be used as a documented language-modeling benchmark/reference source, but its historical bias, dated language, and source-specific rights must remain visible in the manifest. Wikipedia-derived text may be used only with the applicable CC BY-SA attribution and ShareAlike obligations preserved. Project Gutenberg works require per-work and jurisdiction-aware rights review; no blanket public-domain assumption is allowed.

Repository code, scraped web pages, personal communications, private datasets, credential-bearing material, and data with unclear ownership are excluded from training by default unless a separate license and approval record is added. The official PG-19 evaluation split is never mixed into training data.

## Language, domain, and composition policy

The initial policy records target proportions as configuration rather than pretending that a corpus meets them before measurement. Every accepted document receives language and domain labels, and the manifest records measured distributions. English and Amharic are first-class evaluation languages for tokenizer and long-context reporting. General prose, technical/code, and structured/tool text are separate domains. Any imbalance is reported rather than hidden through undocumented resampling.

## Sensitive, unsafe, and low-quality data

Documents containing personal sensitive data, credentials, access tokens, malware payloads, disallowed sexual content, targeted exploitation instructions, or other prohibited material are rejected or quarantined for review. Empty, corrupted, undecodable, spam-like, boilerplate-only, and extremely short documents are rejected with an explicit reason. Exact duplicates and near-duplicates are removed before split assignment. Evaluation examples and their derivatives must not appear in training.

## Retention and deletion

The manifest stores provenance and hashes, not hidden copies of restricted source material. Raw inputs are retained only when the source terms permit retention and the project needs reproducibility. A deletion request or license withdrawal marks the affected source, removes it from rebuild inputs where required, increments the dataset version, and records the replacement or reduced statistics. Derived token shards are deleted or rebuilt when their source eligibility changes.

## Versioning and lineage

A dataset version is immutable after publication. Its manifest records source IDs, URLs or stable identifiers, retrieval timestamps, checksums, license-review status, filtering counts and reasons, tokenizer version, shard checksums, split assignment, and build-tool commit. Every training run and checkpoint must reference exactly one dataset version and the matching tokenizer artifact. A build is reproducible only when the manifest, permitted inputs, tokenizer metadata, code commit, and configuration are all available.

## Required release gate

No dataset is used for a quality claim until source and license review, filtering and leakage checks, split statistics, tokenizer metadata, and shard checksums are complete. A PG-19 or 100M-context result must identify the exact held-out split and must not be described as a general production-quality result without separate evidence.

## References

[1]: https://github.com/google-deepmind/pg19 "Google DeepMind PG-19 repository"

[2]: https://www.tensorflow.org/datasets/catalog/pg19 "TensorFlow Datasets PG-19 documentation"

[3]: https://creativecommons.org/licenses/by-sa/4.0/deed.en "Creative Commons Attribution-ShareAlike 4.0 summary"

[4]: https://www.gutenberg.org/policy/terms_of_use.html "Project Gutenberg Terms of Use"
