from pathlib import Path
import json
import sys


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} MANIFEST_JSON", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"manifest is not valid JSON: {exc}")
    if manifest.get("format") != "attention.dataset_manifest.v1":
        return fail("manifest format is invalid")
    if manifest.get("status") != "policy-defined-not-yet-ingested":
        return fail("manifest status is invalid")
    tokenizer = manifest.get("tokenizer", {})
    expected = {
        "version": "attention.byte_utf8.v1",
        "vocabulary_size": 260,
        "beginning_of_sequence": 256,
        "end_of_sequence": 257,
        "padding": 258,
        "unknown": 259,
    }
    if tokenizer != expected:
        return fail("manifest tokenizer metadata does not match the frozen artifact")
    targets = manifest.get("composition_targets", {})
    for key in ("languages", "domains"):
        values = targets.get(key, {})
        if not values or abs(sum(values.values()) - 1.0) > 1e-12:
            return fail(f"{key} targets must sum to one")
    split_policy = manifest.get("split_policy", {})
    if not split_policy.get("pg19_test_isolation"):
        return fail("PG-19 test isolation is required")
    filtering = manifest.get("filtering", {})
    required_filters = {"exact_duplicates", "near_duplicates", "repeated_boilerplate", "sensitive_data", "unsafe_or_disallowed_content", "corrupted_or_empty_documents"}
    if set(filtering) != required_filters:
        return fail("manifest filtering policy is incomplete")
    if manifest.get("sources") != []:
        return fail("policy template must not claim unreviewed sources")
    print("dataset manifest policy validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
