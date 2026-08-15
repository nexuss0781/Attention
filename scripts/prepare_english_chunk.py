#!/usr/bin/env python3
"""Prepare bounded English train/validation token chunks from JSONL documents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
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
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def record_text(record: dict) -> str:
    for key in ("text", "content", "document", "body"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    if isinstance(record.get("messages"), list):
        parts = []
        for message in record["messages"]:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        if parts:
            return normalize_text("\n".join(parts))
    if isinstance(record.get("message"), str):
        return normalize_text(record["message"])
    return ""


def record_id(record: dict, ordinal: int) -> str:
    for key in ("document_id", "id", "uid", "source_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return f"ordinal-{ordinal:012d}"


def is_english(record: dict, text: str, mode: str, threshold: float) -> bool:
    language = record.get("detected_language", record.get("language"))
    if mode == "dedicated_english":
        return True
    if isinstance(language, str):
        normalized = language.lower().replace("-", "_")
        if normalized in {"en", "eng", "english", "english_code", "en_latin"}:
            return True
        if normalized:
            return False
    score = record.get("language_score", record.get("english_score"))
    if mode == "field":
        return isinstance(score, (int, float)) and score >= threshold
    if isinstance(score, (int, float)):
        return score >= threshold
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(character.isascii() for character in letters)
    return ascii_letters / len(letters) >= threshold


def encode(text: str) -> list[int]:
    return [BOS, *text.encode("utf-8"), EOS]


def write_tokens(path: Path, tokens: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(0, len(tokens), 4096):
            handle.write(" ".join(str(token) for token in tokens[index:index + 4096]))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-train-tokens", required=True)
    parser.add_argument("--output-validation-tokens", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-start", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--language-mode", choices=("dedicated_english", "field", "field_or_heuristic"), default="field_or_heuristic")
    parser.add_argument("--language-threshold", type=float, default=0.98)
    parser.add_argument("--max-train-tokens", type=int, default=0)
    parser.add_argument("--max-validation-tokens", type=int, default=0)
    args = parser.parse_args()

    if min(args.train_start, args.train_count, args.validation_start, args.validation_count) < 0:
        parser.error("document offsets and counts must be nonnegative")
    if args.train_count == 0 or args.validation_count == 0:
        parser.error("train and validation chunks must both be nonempty")
    if not 0.0 <= args.language_threshold <= 1.0:
        parser.error("language threshold must be in [0,1]")
    if args.max_train_tokens < 0 or args.max_validation_tokens < 0:
        parser.error("token limits must be nonnegative")
    source = Path(args.input_jsonl).resolve()
    if not source.is_file():
        parser.error(f"input JSONL does not exist: {source}")

    records = []
    with source.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            text = record_text(record)
            if not text or not is_english(record, text, args.language_mode, args.language_threshold):
                continue
            records.append({"id": record_id(record, ordinal), "text": text})
    records.sort(key=lambda item: item["id"])
    train_end = args.train_start + args.train_count
    validation_end = args.validation_start + args.validation_count
    if train_end > len(records) or validation_end > len(records):
        parser.error(f"requested range exceeds {len(records)} eligible English documents")
    train_records = records[args.train_start:train_end]
    validation_records = records[args.validation_start:validation_end]
    train_ids = {record["id"] for record in train_records}
    validation_ids = {record["id"] for record in validation_records}
    if train_ids & validation_ids:
        parser.error("training and validation document ranges overlap")

    train_tokens = [token for record in train_records for token in encode(record["text"])]
    validation_tokens = [token for record in validation_records for token in encode(record["text"])]
    if args.max_train_tokens:
        train_tokens = train_tokens[:args.max_train_tokens]
    if args.max_validation_tokens:
        validation_tokens = validation_tokens[:args.max_validation_tokens]
    if len(train_tokens) < 2 or len(validation_tokens) < 2:
        parser.error("tokenized train and validation chunks must each contain at least two tokens")
    train_path = Path(args.output_train_tokens).resolve()
    validation_path = Path(args.output_validation_tokens).resolve()
    manifest_path = Path(args.output_manifest).resolve()
    write_tokens(train_path, train_tokens)
    write_tokens(validation_path, validation_tokens)
    manifest = {
        "format": "attention.english_chunk_manifest.v1",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "source_jsonl": str(source),
        "source_sha256": sha256_file(source),
        "tokenizer_version": TOKENIZER_VERSION,
        "vocabulary_size": 260,
        "language": "en",
        "language_mode": args.language_mode,
        "language_threshold": args.language_threshold,
        "max_train_tokens": args.max_train_tokens,
        "max_validation_tokens": args.max_validation_tokens,
        "eligible_document_count": len(records),
        "train": {
            "start": args.train_start,
            "count": len(train_records),
            "document_ids": [record["id"] for record in train_records],
            "token_file": str(train_path),
            "token_sha256": sha256_file(train_path),
            "token_count": len(train_tokens),
        },
        "validation": {
            "start": args.validation_start,
            "count": len(validation_records),
            "document_ids": [record["id"] for record in validation_records],
            "token_file": str(validation_path),
            "token_sha256": sha256_file(validation_path),
            "token_count": len(validation_tokens),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "eligible_documents": len(records),
        "train_documents": len(train_records),
        "validation_documents": len(validation_records),
        "train_tokens": len(train_tokens),
        "validation_tokens": len(validation_tokens),
        "manifest": str(manifest_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
