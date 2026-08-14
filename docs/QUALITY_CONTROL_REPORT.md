# Stage 4.3 Quality-Control Report

The normalized measurement-only documents now pass a deterministic quality-control analyzer. Exact duplicates are grouped by normalized SHA-256, near duplicates are compared using deterministic case-folded five-token shingles and a declared Jaccard threshold, repeated boilerplate is identified from long normalized lines occurring across at least three documents, and split leakage is checked by normalized hash across explicit split labels.

The current four-document measurement set has no exact duplicate groups, no near-duplicate pairs at the 0.80 threshold, no repeated-boilerplate groups, and no split leakage. Its split is `measurement_only`; it is not a training, validation, or test corpus and remains blocked from training approval.

The regression suite uses temporary controlled records to verify that exact duplicates, near duplicates, repeated boilerplate, and cross-split leakage are detected and that the `training_approval` flag remains false when issues exist. These controlled records are tests only and are not part of the corpus.

| Gate | Result |
|---|---:|
| Exact-duplicate detection | Passed |
| Near-duplicate detection | Passed |
| Repeated-boilerplate detection | Passed |
| Split-leakage detection | Passed |
| Training-approval block | Passed |
| Deterministic report rerun | Passed byte-for-byte |
| Full Release suite | 108/108 passed |
| Portable ASAN/UBSAN suite | 108/108 passed |
| Forbidden quadratic-complexity scan | Passed |

The next data-pipeline requirements are safe/disallowed-content filtering, document-quality filtering, source approval, split assignment, and measurable corpus statistics. No production training dataset has been approved.
