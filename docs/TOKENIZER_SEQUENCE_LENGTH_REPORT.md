# Tokenizer Sequence-Length Report

The sequence-length distribution is defined over each non-empty newline-delimited record in the frozen measurement samples. Each record is encoded without BOS or EOS so the reported length is the number of tokenizer IDs consumed by that record itself. Empty lines are counted separately. The report includes mean, nearest-rank p50, nearest-rank p95, and maximum token length.

| Sample | Non-empty lines | Empty lines | Mean tokens | p50 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| English UDHR | 298 | 1 | 54.748322 | 17 | 260 | 557 |
| Amharic Wikipedia Ethiopia | 1,519 | 1 | 39.564187 | 12 | 162 | 1,874 |
| Bilingual concatenation | 1,817 | 2 | 42.054485 | 12 | 194 | 1,874 |
| Repository C++ | 230 | 17 | 46.317391 | 44 | 94 | 115 |
| Structured checkpoint test | 106 | 18 | 51.726415 | 40 | 104 | 118 |

The measurement is descriptive for these source files only. It is not a representative training-corpus sequence-length distribution, and it does not establish model quality, context utilization, or suitability for a production batching policy. The byte-level tokenizer still consumes one token per input byte; therefore, the distribution reflects source-line byte lengths rather than subword segmentation behavior.

The executable writes the canonical CSV to `data/tokenizer_measurement/tokenizer_sequence_lengths.csv` when invoked with the corpus directory, token-measurement output, and sequence-length output paths. A repeated invocation produced byte-identical output.

## Verification

| Gate | Result |
|---|---:|
| Sequence-length measurement executable | Passed |
| Deterministic repeated distribution CSV | Passed byte-for-byte |
| Full Release suite | 106/106 passed |
| Portable ASAN/UBSAN suite | 106/106 passed |
| Forbidden quadratic-complexity scan | Passed |
