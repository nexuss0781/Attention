#!/usr/bin/env python3
"""Prepare the academic Module 1.1 symbol-and-byte regularity lesson."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

BOS = 256
EOS = 257
TOKENIZER_VERSION = "attention.byte_utf8.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return re.sub(r"\s+", " ", value).strip()


def record_text(record: dict) -> str:
    for key in ("text", "content", "document", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    if isinstance(record.get("messages"), list):
        parts = [
            message["content"]
            for message in record["messages"]
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]
        if parts:
            return normalize_text(" ".join(parts))
    if isinstance(record.get("message"), str):
        return normalize_text(record["message"])
    return ""


def record_id(record: dict, ordinal: int) -> str:
    for key in ("document_id", "id", "uid", "source_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return f"ordinal-{ordinal:012d}"


def is_english(record: dict, text: str, threshold: float) -> bool:
    language = record.get("detected_language", record.get("language"))
    if isinstance(language, str):
        normalized = language.lower().replace("-", "_")
        if normalized in {"en", "eng", "english", "english_code", "en_latin"}:
            return True
        if normalized:
            return False
    score = record.get("language_score", record.get("english_score"))
    if isinstance(score, (int, float)):
        return score >= threshold
    letters = [character for character in text if character.isalpha()]
    return bool(letters) and sum(character.isascii() for character in letters) / len(letters) >= threshold


def encode(text: str) -> list[int]:
    return [BOS, *text.encode("utf-8"), EOS]


def write_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for offset in range(0, len(tokens), 4096):
            handle.write(" ".join(str(token) for token in tokens[offset:offset + 4096]) + "\n")


def choose_cases(records: list[dict], case_count: int, prompt_chars: int, continuation_chars: int) -> list[dict]:
    cases: list[dict] = []
    for record in records:
        text = record["text"]
        if len(text) < prompt_chars + continuation_chars:
            continue
        step = max(1, (len(text) - prompt_chars - continuation_chars) // max(case_count, 1))
        for offset in range(0, len(text) - prompt_chars - continuation_chars + 1, step):
            prompt = text[offset:offset + prompt_chars].strip()
            continuation = text[offset + prompt_chars:offset + prompt_chars + continuation_chars].strip()
            if len(prompt) < prompt_chars // 2 or len(continuation) < continuation_chars // 2:
                continue
            if not any(character.isalpha() for character in prompt):
                continue
            cases.append({
                "case_id": f"{record['id']}:{offset}",
                "document_id": record["id"],
                "prompt": prompt,
                "expected_continuation": continuation,
            })
            if len(cases) >= case_count:
                return cases
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-id", default="1.1")
    parser.add_argument("--module-name", default="symbol_and_byte_regularity")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-train-tokens", required=True)
    parser.add_argument("--output-validation-tokens", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-start", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-validation-tokens", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=16)
    parser.add_argument("--prompt-chars", type=int, default=48)
    parser.add_argument("--continuation-chars", type=int, default=32)
    parser.add_argument("--language-threshold", type=float, default=0.98)
    args = parser.parse_args()
    if min(args.train_start, args.train_count, args.validation_start, args.validation_count) < 0:
        parser.error("document offsets and counts must be nonnegative")
    if args.train_count == 0 or args.validation_count == 0 or args.case_count == 0:
        parser.error("train, validation, and case counts must be positive")
    source = Path(args.input_jsonl).resolve()
    if not source.is_file():
        parser.error(f"input JSONL does not exist: {source}")

    records: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            text = record_text(record)
            if text and is_english(record, text, args.language_threshold):
                records.append({"id": record_id(record, ordinal), "text": text})
    records.sort(key=lambda item: item["id"])
    train_records = records[args.train_start:args.train_start + args.train_count]
    validation_records = records[args.validation_start:args.validation_start + args.validation_count]
    if len(train_records) != args.train_count or len(validation_records) != args.validation_count:
        parser.error(f"requested ranges exceed {len(records)} eligible English documents")
    if {item["id"] for item in train_records} & {item["id"] for item in validation_records}:
        parser.error("training and validation documents overlap")

    train_tokens = [token for item in train_records for token in encode(item["text"])]
    validation_tokens = [token for item in validation_records for token in encode(item["text"])]
    if args.max_train_tokens:
        train_tokens = train_tokens[:args.max_train_tokens]
    if args.max_validation_tokens:
        validation_tokens = validation_tokens[:args.max_validation_tokens]
    if len(train_tokens) < 2 or len(validation_tokens) < 2:
        parser.error("train and validation token streams must contain at least two tokens")
    cases = choose_cases(validation_records, args.case_count, args.prompt_chars, args.continuation_chars)
    if len(cases) < args.case_count:
        parser.error(f"only {len(cases)} usable held-out cases could be created")

    train_path = Path(args.output_train_tokens).resolve()
    validation_path = Path(args.output_validation_tokens).resolve()
    cases_path = Path(args.output_cases).resolve()
    manifest_path = Path(args.output_manifest).resolve()
    write_tokens(train_path, train_tokens)
    write_tokens(validation_path, validation_tokens)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("case_id\tdocument_id\tprompt\texpected_continuation\n")
        for case in cases:
            fields = [case["case_id"], case["document_id"], case["prompt"], case["expected_continuation"]]
            handle.write("\t".join(field.replace("\t", " ").replace("\n", " ") for field in fields) + "\n")

    manifest = {
        "format": "attention.module_1_1_lesson_manifest.v1",
        "module_id": args.module_id,
        "module_name": args.module_name,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "source_jsonl": str(source),
        "source_sha256": sha256_file(source),
        "tokenizer_version": TOKENIZER_VERSION,
        "vocabulary_size": 260,
        "language": "en",
        "language_threshold": args.language_threshold,
        "eligible_document_count": len(records),
        "train": {
            "start": args.train_start,
            "count": len(train_records),
            "document_ids": [item["id"] for item in train_records],
            "token_file": str(train_path),
            "token_sha256": sha256_file(train_path),
            "token_count": len(train_tokens),
        },
        "validation": {
            "start": args.validation_start,
            "count": len(validation_records),
            "document_ids": [item["id"] for item in validation_records],
            "token_file": str(validation_path),
            "token_sha256": sha256_file(validation_path),
            "token_count": len(validation_tokens),
        },
        "cases": {
            "file": str(cases_path),
            "sha256": sha256_file(cases_path),
            "count": len(cases),
            "prompt_chars": args.prompt_chars,
            "continuation_chars": args.continuation_chars,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "module_id": args.module_id,
        "eligible_documents": len(records),
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "cases": len(cases),
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
