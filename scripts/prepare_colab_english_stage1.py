#!/usr/bin/env python3
"""Stream a bounded FineWeb English document range and prepare train/validation tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--train-documents", type=int, default=32)
    parser.add_argument("--validation-documents", type=int, default=8)
    parser.add_argument("--max-train-tokens", type=int, default=4096)
    parser.add_argument("--max-validation-tokens", type=int, default=1024)
    parser.add_argument("--module", choices=("module_1_1",), default="module_1_1")
    args = parser.parse_args()
    if min(args.start_document, args.train_documents, args.validation_documents,
           args.max_train_tokens, args.max_validation_tokens) < 0:
        parser.error("document offsets, counts, and token limits must be nonnegative")
    if args.train_documents == 0 or args.validation_documents == 0:
        parser.error("train and validation document counts must be positive")

    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        print("missing dependency: install with `pip install -q datasets huggingface_hub`", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_documents.jsonl"
    try:
        info = HfApi().dataset_info(args.dataset, revision=args.revision)
        resolved_revision = getattr(info, "sha", None) or args.revision
    except Exception as exc:  # network/auth failure is reported with context
        print(f"could not resolve dataset revision: {exc}", file=sys.stderr)
        return 1

    try:
        stream = load_dataset(
            args.dataset,
            name=args.config,
            split="train",
            revision=resolved_revision,
            streaming=True,
        )
        stream = stream.skip(args.start_document)
        selected = []
        target = args.train_documents + args.validation_documents
        for source_index, record in enumerate(stream, start=args.start_document):
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            selected.append({
                "document_id": f"fineweb-{source_index:012d}",
                "source_row_index": source_index,
                "detected_language": "english",
                "text": text,
            })
            if len(selected) >= target:
                break
    except Exception as exc:
        print(f"streaming dataset read failed: {exc}", file=sys.stderr)
        return 1

    if len(selected) != target:
        print(f"stream ended after {len(selected)} eligible documents; required {target}", file=sys.stderr)
        return 1
    with selected_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    train_path = output_dir / "train.tokens"
    validation_path = output_dir / "validation.tokens"
    chunk_manifest = output_dir / "chunk_manifest.json"
    prepare = Path(__file__).with_name("prepare_english_chunk.py")
    command = [
        sys.executable,
        str(prepare),
        "--input-jsonl", str(selected_path),
        "--output-train-tokens", str(train_path),
        "--output-validation-tokens", str(validation_path),
        "--output-manifest", str(chunk_manifest),
        "--dataset-id", "fineweb_english",
        "--dataset-revision", resolved_revision,
        "--train-start", "0",
        "--train-count", str(args.train_documents),
        "--validation-start", str(args.train_documents),
        "--validation-count", str(args.validation_documents),
        "--language-mode", "dedicated_english",
        "--max-train-tokens", str(args.max_train_tokens),
        "--max-validation-tokens", str(args.max_validation_tokens),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    module_prepare = Path(__file__).with_name("prepare_module_1_1.py")
    module_command = [
        sys.executable,
        str(module_prepare),
        "--input-jsonl", str(selected_path),
        "--output-train-tokens", str(output_dir / "module_1_1_train.tokens"),
        "--output-validation-tokens", str(output_dir / "module_1_1_validation.tokens"),
        "--output-cases", str(output_dir / "module_1_1_cases.tsv"),
        "--output-manifest", str(output_dir / "module_1_1_lesson_manifest.json"),
        "--dataset-id", "fineweb_english",
        "--dataset-revision", resolved_revision,
        "--train-start", "0",
        "--train-count", str(args.train_documents),
        "--validation-start", str(args.train_documents),
        "--validation-count", str(args.validation_documents),
        "--max-train-tokens", str(args.max_train_tokens),
        "--max-validation-tokens", str(args.max_validation_tokens),
    ]
    module_completed = subprocess.run(module_command, check=False)
    if module_completed.returncode != 0:
        return module_completed.returncode
    metadata = {
        "format": "attention.colab_dataset_preparation.v1",
        "dataset": args.dataset,
        "config": args.config,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "start_document": args.start_document,
        "train_documents": args.train_documents,
        "validation_documents": args.validation_documents,
        "max_train_tokens": args.max_train_tokens,
        "max_validation_tokens": args.max_validation_tokens,
        "selected_documents": str(selected_path),
                    "chunk_manifest": str(chunk_manifest),
            "module": args.module,
            "module_1_1_lesson_manifest": str(output_dir / "module_1_1_lesson_manifest.json"),
            "module_1_1_cases": str(output_dir / "module_1_1_cases.tsv"),

    }
    (output_dir / "download_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
