from hashlib import sha256
from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "scripts" / "quality_control.py"


def row(document_id: str, text: str, split: str) -> dict:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return {
        "document_id": document_id,
        "status": "accepted_measurement_only",
        "normalized_sha256": sha256(normalized.encode("utf-8")).hexdigest(),
        "text": normalized,
        "split": split,
    }


def main() -> int:
    shared = "Copyright and licensing notice repeated for quality-control testing purposes."
    base = "one two three four five six seven eight nine ten "
    rows = [
        row("a", shared + "\n" + base, "train"),
        row("b", shared + "\n" + base, "validation"),
        row("c", shared + "\n" + base + " eleven", "test"),
    ]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        source = work / "documents.jsonl"
        output = work / "quality.json"
        source.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        result = subprocess.run([sys.executable, str(QUALITY), str(source), str(output)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["training_approval"] is False
        assert report["exact_duplicate_document_count"] == 2
        assert report["near_duplicate_pairs"]
        assert report["repeated_boilerplate_groups"]
        assert report["split_leakage"]
        assert report["checks"]["exact_duplicates_removed_before_training"] is False
        assert report["checks"]["split_leakage_absent"] is False
    print("quality-control duplicate, boilerplate, leakage, and approval-block tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
