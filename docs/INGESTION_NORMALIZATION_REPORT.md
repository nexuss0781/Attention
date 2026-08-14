# Stage 4.2 Ingestion and Normalization Report

The first ingestion slice is deliberately **measurement-only**. It consumes only the four local/public samples listed in `data/ingestion_measurement_manifest_v1.json`; every source is marked `approved_for_training: false` until separate license and corpus approval is completed.

` scripts/ingest_sources.py` reads the manifest, verifies each input SHA-256 checksum, decodes strict UTF-8, converts CRLF/CR line endings to LF, applies Unicode NFC normalization, rejects invalid or empty documents, detects a conservative language label, and emits normalized JSONL documents with source URL, owner/provider, license status, intended use, source and normalized hashes, language/domain labels, and document IDs. It also writes inspectable preview samples and deterministic summary counts.

The current run accepted four documents and rejected none. The output remains explicitly `measurement-only-not-training-approved`; it is not a training corpus and does not support a model-quality claim.

| Gate | Result |
|---|---:|
| Manifest-driven ingestion | Passed |
| SHA-256 source-lineage verification | Passed |
| Strict UTF-8 rejection test | Passed |
| Empty-document rejection test | Passed |
| Checksum-mismatch rejection test | Passed |
| LF/NFC normalization and provenance retention | Passed |
| Deterministic JSONL and summary rerun | Passed byte-for-byte |
| Full Release suite | 108/108 passed |
| Portable ASAN/UBSAN suite | 108/108 passed |
| Forbidden quadratic-complexity scan | Passed |

The next ingestion work remains pending: language detection beyond the conservative script heuristic, duplicate and near-duplicate filtering, boilerplate/spam/unsafe-content controls, split assignment, and approved training-source population.
