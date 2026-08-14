# Tokenizer Measurement Report

The `attention.byte_utf8.v1` measurement harness evaluates five transparent samples: English, Amharic, bilingual English/Amharic, C++ source, and structured checkpoint-test text. The English sample is fetched from the United Nations Universal Declaration of Human Rights page [1]. The Amharic sample is fetched from the Amharic Wikipedia Ethiopia article [2]. The official OHCHR Amharic translation PDF is retained in the corpus provenance, but its custom font encoding is not recoverable as Unicode by the available text extractor; it is therefore not silently treated as usable text [3]. Code and structured samples are copied from this repository and are clearly labeled as implementation-derived samples.

The harness reports raw UTF-8 bytes, Unicode code points, byte-level token count, bytes per token, unknown-token count and rate, and exact decode round-trip status. Since the tokenizer maps each valid input byte to one token, the observed one byte per token and zero unknown rate are construction properties, not evidence of semantic quality, compression, or language-model performance.

| Sample | UTF-8 bytes | Code points | Tokens | Bytes/token | Unknown rate | Round trip |
|---|---:|---:|---:|---:|---:|---|
| English UDHR | 16,613 | 16,552 | 16,613 | 1.000000 | 0.000000 | Pass |
| Amharic Wikipedia Ethiopia | 61,617 | 26,643 | 61,617 | 1.000000 | 0.000000 | Pass |
| Bilingual concatenation | 78,231 | 43,196 | 78,231 | 1.000000 | 0.000000 | Pass |
| Repository C++ | 10,899 | 10,899 | 10,899 | 1.000000 | 0.000000 | Pass |
| Structured checkpoint test | 5,606 | 5,606 | 5,606 | 1.000000 | 0.000000 | Pass |

The measurement command is reproducible through `attention_tokenizer_measurements data/tokenizer_measurement/corpus data/tokenizer_measurement/tokenizer_measurements.csv`. A second run produced byte-identical CSV output. The corpus manifest and source records are stored under `data/tokenizer_measurement/`.

These results complete the requested sample measurements, but they do not complete the final tokenizer-selection gate. Subword algorithm selection, production normalization policy, representative corpus statistics, sequence-length distribution, training-run tokenizer freezing, and checkpoint metadata integration remain future work.

## Verification

| Gate | Result |
|---|---:|
| Measurement executable | Passed |
| Deterministic repeated CSV | Passed |
| Full Release suite | 106/106 passed |
| Portable ASAN/UBSAN suite | 106/106 passed |
| Forbidden quadratic-complexity scan | Passed |

## References

[1]: https://www.un.org/en/about-us/universal-declaration-of-human-rights "United Nations: Universal Declaration of Human Rights"

[2]: https://am.wikipedia.org/wiki/%E1%8A%A2%E1%89%B5%E1%8B%AE%E1%8C%B5%E1%8B%AB "Amharic Wikipedia: Ethiopia"

[3]: https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/amh.pdf "OHCHR: Universal Declaration of Human Rights, Amharic PDF"
