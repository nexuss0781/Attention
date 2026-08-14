from pathlib import Path
import json
import sys


def build_manifest(run_id: str) -> dict:
    return {
        "format": "attention.training_run_manifest.v1",
        "run_id": run_id,
        "tokenizer": {
            "version": "attention.byte_utf8.v1",
            "vocabulary_size": 260,
            "beginning_of_sequence": 256,
            "end_of_sequence": 257,
            "padding": 258,
            "unknown": 259,
        },
        "checkpoint_format": "attention.checkpoint.v2",
        "tokenizer_metadata_required_on_reload": True,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} RUN_ID OUTPUT_JSON", file=sys.stderr)
        return 2
    run_id, output_name = sys.argv[1:]
    if not run_id or any(char in run_id for char in "\r\n"):
        print("run ID must be nonempty and single-line", file=sys.stderr)
        return 2
    output = Path(output_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_manifest(run_id), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
