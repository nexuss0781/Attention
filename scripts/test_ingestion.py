from pathlib import Path
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / "scripts" / "ingest_sources.py"
BASE = json.loads((ROOT / "data" / "ingestion_measurement_manifest_v1.json").read_text(encoding="utf-8"))


def run_case(manifest: dict, source_bytes: bytes, expected_reason: str | None) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        case = json.loads(json.dumps(manifest))
        # The ingester resolves repository-relative paths, so write the temporary source under ROOT.
        repo_source = ROOT / "data" / "_ingestion_test_source.bin"
        repo_source.write_bytes(source_bytes)
        case["sources"] = [dict(case["sources"][0], relative_path=str(repo_source.relative_to(ROOT)), sha256="0" * 64)]
        manifest_path = work / "manifest.json"
        manifest_path.write_text(json.dumps(case), encoding="utf-8")
        output = work / "documents.jsonl"
        samples = work / "samples.jsonl"
        summary = work / "summary.json"
        result = subprocess.run([sys.executable, str(INGEST), str(manifest_path), str(output), str(samples), str(summary)], capture_output=True, text=True)
        try:
            if expected_reason is None:
                assert result.returncode == 0, result.stderr
                parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                assert len(parsed) == 1
                assert parsed[0]["status"] == "accepted_measurement_only"
                assert parsed[0]["permitted_use"] == "measurement_only"
                assert parsed[0]["input_sha256"] == "0" * 64
                return json.loads(summary.read_text(encoding="utf-8"))
            assert result.returncode == 1, result.stderr
            parsed = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            assert parsed[0]["rejection_reason"] == expected_reason
            return json.loads(summary.read_text(encoding="utf-8"))
        finally:
            repo_source.unlink(missing_ok=True)


def main() -> int:
    # The source checksum mismatch is rejected before decoding.
    run_case(BASE, b"valid text\n", "source_checksum_mismatch")
    # An exact checksum is required for the decode/empty checks below.
    import hashlib
    base = json.loads(json.dumps(BASE))
    repo_source = ROOT / "data" / "_ingestion_test_source.bin"
    try:
        for payload, reason in ((b"\xff", "invalid_utf8"), (b"\n\r\n", "empty_document")):
            repo_source.write_bytes(payload)
            case = json.loads(json.dumps(base))
            case["sources"] = [dict(case["sources"][0], relative_path=str(repo_source.relative_to(ROOT)), sha256=hashlib.sha256(payload).hexdigest())]
            with tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                manifest = work / "manifest.json"
                manifest.write_text(json.dumps(case), encoding="utf-8")
                result = subprocess.run([sys.executable, str(INGEST), str(manifest), str(work / "docs.jsonl"), str(work / "samples.jsonl"), str(work / "summary.json")], capture_output=True, text=True)
                assert result.returncode == 1, result.stderr
                rows = [json.loads(line) for line in (work / "docs.jsonl").read_text(encoding="utf-8").splitlines()]
                assert rows[0]["rejection_reason"] == reason
    finally:
        repo_source.unlink(missing_ok=True)
    print("ingestion rejection, provenance, and policy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
