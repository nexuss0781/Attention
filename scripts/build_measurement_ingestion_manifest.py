from hashlib import sha256
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DEFINITIONS = [
    {
        "source_id": "un_udhr_english",
        "relative_path": "data/tokenizer_measurement/corpus/english_udhr.txt",
        "source_url": "https://www.un.org/en/about-us/universal-declaration-of-human-rights",
        "owner_provider": "United Nations",
        "license_status": "source-specific-review-required",
        "permitted_use": "measurement_only",
        "approved_for_training": False,
        "language": "english",
        "domain": "general_prose",
    },
    {
        "source_id": "am_wikipedia_ethiopia",
        "relative_path": "data/tokenizer_measurement/corpus/amharic_wikipedia_ethiopia.txt",
        "source_url": "https://am.wikipedia.org/wiki/%E1%8A%A2%E1%89%B5%E1%8B%AE%E1%8C%B5%E1%8B%AB",
        "owner_provider": "Wikimedia contributors",
        "license_status": "CC-BY-SA-4.0-attribution-required",
        "permitted_use": "measurement_only",
        "approved_for_training": False,
        "language": "amharic",
        "domain": "general_prose",
    },
    {
        "source_id": "attention_cpp_internal",
        "relative_path": "data/tokenizer_measurement/corpus/code_cpp.txt",
        "source_url": "repository://Attention/src/transformer_model.cpp",
        "owner_provider": "Attention repository contributors",
        "license_status": "repository-license-review-required",
        "permitted_use": "measurement_only",
        "approved_for_training": False,
        "language": "english_code",
        "domain": "technical_code",
    },
    {
        "source_id": "attention_structured_checkpoint_test",
        "relative_path": "data/tokenizer_measurement/corpus/structured_checkpoint_test.txt",
        "source_url": "repository://Attention/tests/test_checkpoint.cpp",
        "owner_provider": "Attention repository contributors",
        "license_status": "repository-license-review-required",
        "permitted_use": "measurement_only",
        "approved_for_training": False,
        "language": "english_code",
        "domain": "structured_tool_text",
    },
]


def main() -> int:
    output = ROOT / "data" / "ingestion_measurement_manifest_v1.json"
    if len(sys.argv) == 2:
        output = Path(sys.argv[1])
    records = []
    for item in SOURCE_DEFINITIONS:
        path = ROOT / item["relative_path"]
        if not path.is_file():
            print(f"missing source: {path}", file=sys.stderr)
            return 1
        raw = path.read_bytes()
        record = dict(item)
        record["sha256"] = sha256(raw).hexdigest()
        record["retrieved_or_built_at"] = "2026-08-14"
        records.append(record)
    manifest = {
        "format": "attention.ingestion_manifest.v1",
        "dataset_id": "attention-measurement-ingestion-v1",
        "status": "measurement-only-not-training-approved",
        "tokenizer_version": "attention.byte_utf8.v1",
        "sources": records,
        "normalization": {
            "encoding": "UTF-8-strict",
            "unicode_normalization": "NFC",
            "line_endings": "LF",
            "empty_documents": "reject",
            "invalid_utf8": "reject",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
