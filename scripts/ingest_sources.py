from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]


def normalize_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    return text.strip("\n")


def detect_language(text: str, declared: str, domain: str) -> str:
    if any(0x1200 <= ord(char) <= 0x137F for char in text):
        return "amharic"
    if domain in {"technical_code", "structured_tool_text"}:
        return "english_code"
    return declared if declared else "unknown"


def reject(reason: str, record: dict) -> dict:
    output = dict(record)
    output.update({"status": "rejected", "rejection_reason": reason})
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("samples_jsonl", type=Path)
    parser.add_argument("summary_json", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "attention.ingestion_manifest.v1":
        print("unsupported ingestion manifest", file=sys.stderr)
        return 2
    if manifest.get("status") != "measurement-only-not-training-approved":
        print("this first ingestion slice requires measurement-only status", file=sys.stderr)
        return 2
    accepted = []
    rejected = []
    samples = []
    for source in sorted(manifest.get("sources", []), key=lambda item: item["source_id"]):
        path = ROOT / source["relative_path"]
        base = {
            "source_id": source["source_id"],
            "source_url": source["source_url"],
            "owner_provider": source["owner_provider"],
            "license_status": source["license_status"],
            "permitted_use": source["permitted_use"],
            "approved_for_training": bool(source["approved_for_training"]),
            "declared_language": source["language"],
            "domain": source["domain"],
            "input_sha256": source["sha256"],
            "normalization": manifest["normalization"],
        }
        if not path.is_file():
            rejected.append(reject("missing_source", base))
            continue
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != source["sha256"]:
            rejected.append(reject("source_checksum_mismatch", base))
            continue
        try:
            normalized = normalize_text(raw)
        except UnicodeDecodeError:
            rejected.append(reject("invalid_utf8", base))
            continue
        if not normalized.strip():
            rejected.append(reject("empty_document", base))
            continue
        normalized_bytes = normalized.encode("utf-8")
        document = dict(base)
        document.update({
            "document_id": sha256((source["source_id"] + "\n" + sha256(normalized_bytes).hexdigest()).encode("utf-8")).hexdigest(),
            "status": "accepted_measurement_only",
            "detected_language": detect_language(normalized, source["language"], source["domain"]),
            "normalized_sha256": sha256(normalized_bytes).hexdigest(),
            "normalized_bytes": len(normalized_bytes),
            "line_count": normalized.count("\n") + 1,
            "text": normalized,
        })
        accepted.append(document)
        samples.append({
            "document_id": document["document_id"],
            "source_id": document["source_id"],
            "detected_language": document["detected_language"],
            "domain": document["domain"],
            "preview": normalized[:240],
        })
    accepted.sort(key=lambda item: item["document_id"])
    rejected.sort(key=lambda item: (item["source_id"], item["rejection_reason"]))
    samples.sort(key=lambda item: item["document_id"])
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as output:
        for item in accepted:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        for item in rejected:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with args.samples_jsonl.open("w", encoding="utf-8", newline="\n") as output:
        for item in samples:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "format": "attention.ingestion_summary.v1",
        "dataset_id": manifest["dataset_id"],
        "status": "measurement-only-not-training-approved",
        "tokenizer_version": manifest["tokenizer_version"],
        "accepted_documents": len(accepted),
        "rejected_documents": len(rejected),
        "rejection_counts": {},
        "language_counts": {},
        "domain_counts": {},
    }
    for item in rejected:
        reason = item["rejection_reason"]
        summary["rejection_counts"][reason] = summary["rejection_counts"].get(reason, 0) + 1
    for item in accepted:
        language = item["detected_language"]
        domain = item["domain"]
        summary["language_counts"][language] = summary["language_counts"].get(language, 0) + 1
        summary["domain_counts"][domain] = summary["domain_counts"].get(domain, 0) + 1
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
