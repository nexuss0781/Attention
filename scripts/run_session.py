#!/usr/bin/env python3
"""Execute one competency-training session from its immutable session manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required file does not exist: {path}")
    return path


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--train-tokens", required=True)
    parser.add_argument("--validation-tokens", required=True)
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    manifest_path = require(session_dir / "session_manifest.json")
    manifest = load_json(manifest_path)
    if manifest.get("state") != "PLANNED":
        raise RuntimeError(f"session cannot start from state {manifest.get('state')}; create a new session ID for retry")
    executable = require(Path(args.executable).resolve())
    train_tokens = require(Path(args.train_tokens).resolve())
    validation_tokens = require(Path(args.validation_tokens).resolve())
    parent = require(Path(manifest["parent_checkpoint"]))

    training_log = session_dir / "training_log.json"
    state_checkpoint = session_dir / "training_state.checkpoint"
    model_checkpoint = session_dir / "model.checkpoint"
    trace = session_dir / "training_trace.csv"
    validation_report = session_dir / "validation_report.json"
    environment = os.environ.copy()
    environment.update(
        {
            "ATTENTION_SESSION_ID": manifest["session_id"],
            "ATTENTION_STAGE_ID": manifest["stage_id"],
            "ATTENTION_DATASET_ID": manifest["dataset_id"],
            "ATTENTION_DATASET_REVISION": manifest.get("dataset_revision", "session-manifest"),
            "ATTENTION_CODE_COMMIT": environment.get("ATTENTION_CODE_COMMIT", "unknown"),
            "ATTENTION_MAX_STEPS": str(manifest["max_steps"]),
        }
    )
    command = [
        str(executable),
        str(train_tokens),
        str(training_log),
        str(state_checkpoint),
        str(validation_tokens),
        str(model_checkpoint),
        str(parent),
    ]
    with trace.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        manifest["state"] = "FAILED"
        manifest["failure"] = "training executable returned nonzero"
        atomic_json(manifest_path, manifest)
        (session_dir / "STATE").write_text("FAILED\n", encoding="utf-8")
        raise RuntimeError(f"training executable failed with exit code {completed.returncode}; see {trace}")

    run_log = load_json(training_log)
    records = run_log.get("records", [])
    if not records:
        raise RuntimeError("training run log contains no records")
    required_fields = {"loss_before", "loss_after", "validation_loss", "gradient_l2_norm"}
    if any(not required_fields.issubset(record) for record in records):
        raise RuntimeError("training run log is missing required competency metrics")
    if any(not finite(record[field]) for record in records for field in required_fields):
        raise RuntimeError("training run contains nonfinite competency metrics")
    if not float(records[-1]["loss_after"]) < float(records[0]["loss_before"]):
        raise RuntimeError("training loss did not improve over the session")

    stdout_lines = trace.read_text(encoding="utf-8").splitlines()
    values = {}
    for line in stdout_lines:
        if "," in line:
            key, value = line.split(",", 1)
            if key in {"reloaded_loss", "resumed_loss", "validation_loss"}:
                values[key] = float(value)
    if not finite(values.get("reloaded_loss")) or not finite(values.get("resumed_loss")):
        raise RuntimeError("checkpoint reload losses are missing or nonfinite")
    if values["reloaded_loss"] != values["resumed_loss"]:
        raise RuntimeError("model and training-state reload losses differ")

    report = {
        "format": "attention.competency_validation_report.v1",
        "session_id": manifest["session_id"],
        "competency_id": manifest["competency_id"],
        "automated_pass": True,
        "training_loss_before": float(records[0]["loss_before"]),
        "training_loss_after": float(records[-1]["loss_after"]),
        "validation_loss_first": float(records[0]["validation_loss"]),
        "validation_loss_final": float(records[-1]["validation_loss"]),
        "validation_loss_finite": all(finite(record["validation_loss"]) for record in records),
        "gradient_norms_finite": all(finite(record["gradient_l2_norm"]) for record in records),
        "model_reload_loss": values["reloaded_loss"],
        "training_state_reload_loss": values["resumed_loss"],
        "manual_review_required": True,
    }
    atomic_json(validation_report, report)
    manifest["state"] = "VALIDATING"
    manifest["automated_validation_passed"] = True
    atomic_json(manifest_path, manifest)
    (session_dir / "STATE").write_text("VALIDATING\n", encoding="utf-8")
    print(f"automated competency validation passed: {session_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"session run error: {exc}", file=sys.stderr)
        raise SystemExit(1)
