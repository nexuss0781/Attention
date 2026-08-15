#!/usr/bin/env python3
"""Create and seal immutable competency-training session artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import tempfile

FORMAT = "attention.session_manifest.v1"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,127}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_file(path_text: str, description: str) -> Path:
    path = Path(path_text).resolve()
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    return path


def session_dir(root: Path, session_id: str) -> Path:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("session ID must be 3-128 characters of letters, digits, '.', '_' or '-'")
    return root.resolve() / session_id


def load_manifest(directory: Path) -> dict:
    path = directory / "session_manifest.json"
    if not path.is_file():
        raise ValueError(f"session manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("session manifest format is invalid")
    return manifest


def init_session(args: argparse.Namespace) -> int:
    root = Path(args.root)
    directory = session_dir(root, args.session_id)
    parent = require_file(args.parent_checkpoint, "parent checkpoint")
    dataset_manifest = require_file(args.dataset_manifest, "dataset manifest")
    dataset_chunk = require_file(args.dataset_chunk, "dataset chunk")
    if args.max_steps <= 0 or args.max_tokens <= 0 or args.seed < 0:
        raise ValueError("seed, max_steps, and max_tokens must be valid")
    try:
        chunk_metadata = json.loads(dataset_chunk.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        chunk_metadata = {}
    dataset_id = chunk_metadata.get("dataset_id", "unknown_dataset")
    dataset_revision = chunk_metadata.get("dataset_revision", "unknown_revision")
    if not isinstance(dataset_id, str) or not isinstance(dataset_revision, str):
        raise ValueError("dataset chunk metadata identifiers are invalid")

    existing = directory / "session_manifest.json"
    if existing.exists():
        manifest = load_manifest(directory)
        expected = {
            "session_id": args.session_id,
            "stage_id": args.stage_id,
            "competency_id": args.competency_id,
            "parent_checkpoint_sha256": sha256_file(parent),
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "dataset_chunk_sha256": sha256_file(dataset_chunk),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"existing session input mismatch for {key}")
        print(f"session already initialized: {directory}")
        return 0

    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"refusing to initialize nonempty session directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": FORMAT,
        "session_id": args.session_id,
        "state": "PLANNED",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "stage_id": args.stage_id,
        "competency_id": args.competency_id,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": sha256_file(parent),
        "dataset_manifest": str(dataset_manifest),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
                    "dataset_chunk": str(dataset_chunk),
            "dataset_chunk_sha256": sha256_file(dataset_chunk),
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,

        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "automatic_promotion": False,
        "manual_review_required": True,
        "artifact_directory": str(directory),
    }
    write_json(existing, manifest)
    atomic_write(
        directory / "manual_decision.template.txt",
        "DECISION: PROMOTE | RETRY | ABORT\n"
        f"COMPETENCY: {args.competency_id}\n"
        "RATIONALE: <human assessment>\n"
        "NEXT_ACTION: <next competency or retry plan>\n",
    )
    atomic_write(directory / "STATE", "PLANNED\n")
    print(f"initialized session: {directory}")
    return 0


def seal_session(args: argparse.Namespace) -> int:
    directory = Path(args.session_dir).resolve()
    manifest = load_manifest(directory)
    if manifest["state"] not in {"PLANNED", "RUNNING", "VALIDATING"}:
        raise ValueError(f"cannot seal session in state {manifest['state']}")
    required = [
        "training_log.json",
        "validation_report.json",
        "model.checkpoint",
        "training_state.checkpoint",
    ]
    for name in required:
        require_file(str(directory / name), f"session artifact {name}")
    files = {}
    excluded = {"artifact_manifest.json", "manual_decision.txt", "manual_decision.template.txt"}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name not in excluded and path.name != "session_manifest.json":
            files[str(path.relative_to(directory))] = sha256_file(path)
    artifact_manifest = {
        "format": "attention.session_artifacts.v1",
        "session_id": manifest["session_id"],
        "session_manifest_sha256": sha256_file(directory / "session_manifest.json"),
        "files": files,
    }
    write_json(directory / "artifact_manifest.json", artifact_manifest)
    manifest["state"] = "AWAITING_REVIEW"
    manifest["automated_validation_passed"] = True
    manifest["artifact_manifest"] = str(directory / "artifact_manifest.json")
    write_json(directory / "session_manifest.json", manifest)
    atomic_write(directory / "STATE", "AWAITING_REVIEW\n")
    print(f"session sealed for manual review: {directory}")
    return 0


def record_decision(args: argparse.Namespace) -> int:
    directory = Path(args.session_dir).resolve()
    manifest = load_manifest(directory)
    if manifest["state"] != "AWAITING_REVIEW":
        raise ValueError(f"manual decision requires AWAITING_REVIEW, got {manifest['state']}")
    decision_path = require_file(args.decision_file, "manual decision file")
    lines = {}
    for raw_line in decision_path.read_text(encoding="utf-8").splitlines():
        if ":" in raw_line:
            key, value = raw_line.split(":", 1)
            lines[key.strip().upper()] = value.strip()
    decision = lines.get("DECISION", "")
    if decision not in {"PROMOTE", "RETRY", "ABORT"}:
        raise ValueError("DECISION must be PROMOTE, RETRY, or ABORT")
    if lines.get("COMPETENCY") != manifest["competency_id"]:
        raise ValueError("manual decision competency does not match session")
    if not lines.get("RATIONALE") or lines["RATIONALE"].startswith("<"):
        raise ValueError("manual decision requires a rationale")
    if not lines.get("NEXT_ACTION") or lines["NEXT_ACTION"].startswith("<"):
        raise ValueError("manual decision requires a next action")
    atomic_write(directory / "manual_decision.txt", decision_path.read_text(encoding="utf-8"))
    state = {"PROMOTE": "PROMOTED", "RETRY": "RETRY", "ABORT": "ABORTED"}[decision]
    manifest["state"] = state
    manifest["manual_decision"] = decision
    write_json(directory / "session_manifest.json", manifest)
    atomic_write(directory / "STATE", state + "\n")
    print(f"session decision recorded: {state}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--session-id", required=True)
    init.add_argument("--stage-id", required=True)
    init.add_argument("--competency-id", required=True)
    init.add_argument("--parent-checkpoint", required=True)
    init.add_argument("--dataset-manifest", required=True)
    init.add_argument("--dataset-chunk", required=True)
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--max-steps", type=int, required=True)
    init.add_argument("--max-tokens", type=int, required=True)
    init.set_defaults(function=init_session)

    seal = commands.add_parser("seal")
    seal.add_argument("--session-dir", required=True)
    seal.set_defaults(function=seal_session)

    decision = commands.add_parser("decision")
    decision.add_argument("--session-dir", required=True)
    decision.add_argument("--decision-file", required=True)
    decision.set_defaults(function=record_decision)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"session artifact error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
