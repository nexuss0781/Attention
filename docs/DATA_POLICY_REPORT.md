# Stage 4 Data-Policy Report

The first Stage 4 slice is complete at the **policy-definition** level. `docs/DATA_POLICY.md` defines permitted-source review, ownership and license recording, excluded sources, sensitive-data handling, language/domain targets, retention and deletion behavior, split isolation, and immutable dataset versioning. `data/dataset_manifest_v1.json` is a policy-defined manifest template, not an ingested corpus claim.

The declared initial composition targets are 70% English, 10% Amharic, and 20% other languages; and 75% general prose, 15% technical/code, and 10% structured/tool text. These are targets only. The eventual ingestion pipeline must report measured distributions and deviations rather than silently asserting that the targets were met.

The manifest freezes `attention.byte_utf8.v1` metadata, requires PG-19 test isolation, requires filtering counts and reasons, and leaves the source list empty until each source has a documented identifier, owner/provider, license review, retrieval timestamp, checksum, intended split, and permitted-use decision. This avoids treating public availability or historical public-domain status as a blanket training permission.

The saved validator `scripts/validate_dataset_manifest.py` checks the manifest format, frozen tokenizer metadata, composition sums, PG-19 isolation, filtering-policy completeness, and the deliberate absence of unreviewed sources.

## Verification

| Gate | Result |
|---|---:|
| Manifest JSON validation | Passed |
| Declared language/domain target sums | Passed |
| Frozen tokenizer identity match | Passed |
| PG-19 test isolation field | Passed |
| Empty unreviewed-source list | Passed |
| Deterministic validator output | Passed |
| Quadratic-complexity impact | None; no model/runtime code changed |

## References

[1]: https://github.com/google-deepmind/pg19 "Google DeepMind PG-19 repository"

[2]: https://www.tensorflow.org/datasets/catalog/pg19 "TensorFlow Datasets PG-19 documentation"

[3]: https://creativecommons.org/licenses/by-sa/4.0/deed.en "Creative Commons Attribution-ShareAlike 4.0 summary"

[4]: https://www.gutenberg.org/policy/terms_of_use.html "Project Gutenberg Terms of Use"
