from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import json
import re
import sys


def normalized_words(text: str) -> list[str]:
    return re.findall(r"\S+", text.casefold())


def shingles(words: list[str], width: int = 5) -> set[str]:
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index:index + width]) for index in range(len(words) - width + 1)}


def similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 < args.near_threshold <= 1.0:
        print("near threshold must be in (0, 1]", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in args.documents.read_text(encoding="utf-8").splitlines() if line]
    accepted = [row for row in rows if row.get("status") == "accepted_measurement_only"]
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_split: dict[str, list[str]] = defaultdict(list)
    word_sets: dict[str, set[str]] = {}
    for row in accepted:
        document_id = row["document_id"]
        by_hash[row["normalized_sha256"]].append(document_id)
        split = row.get("split", "measurement_only")
        by_split[split].append(document_id)
        word_sets[document_id] = shingles(normalized_words(row["text"]))
    exact_groups = [sorted(ids) for ids in by_hash.values() if len(ids) > 1]
    near_pairs = []
    ids = sorted(word_sets)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            score = similarity(word_sets[left_id], word_sets[right_id])
            if score >= args.near_threshold:
                near_pairs.append({"left": left_id, "right": right_id, "jaccard": round(score, 12)})
    line_counts: Counter[str] = Counter()
    line_documents: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        for line in row["text"].split("\n"):
            normalized_line = " ".join(line.split())
            if len(normalized_line) >= 40:
                key = sha256(normalized_line.casefold().encode("utf-8")).hexdigest()
                line_counts[key] += 1
                line_documents[key].add(row["document_id"])
    boilerplate = sorted(
        {key: {"occurrences": line_counts[key], "documents": sorted(line_documents[key])} for key in line_counts if len(line_documents[key]) >= 3}.items()
    )
    split_leakage = []
    for digest, document_ids in by_hash.items():
        splits = {row.get("split", "measurement_only") for row in accepted if row["normalized_sha256"] == digest}
        if len(splits) > 1:
            split_leakage.append({"normalized_sha256": digest, "splits": sorted(splits), "documents": sorted(document_ids)})
    report = {
        "format": "attention.quality_control_report.v1",
        "status": "measurement-only-not-training-approved",
        "training_approval": False,
        "documents_seen": len(rows),
        "accepted_documents": len(accepted),
        "split_counts": {key: len(sorted(value)) for key, value in sorted(by_split.items())},
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_document_count": sum(len(group) for group in exact_groups),
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_threshold": args.near_threshold,
        "repeated_boilerplate_groups": [entry[1] | {"line_hash": entry[0]} for entry in boilerplate],
        "split_leakage": split_leakage,
        "checks": {
            "exact_duplicates_removed_before_training": len(exact_groups) == 0,
            "near_duplicates_removed_before_training": len(near_pairs) == 0,
            "repeated_boilerplate_reviewed": len(boilerplate) == 0,
            "split_leakage_absent": len(split_leakage) == 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
